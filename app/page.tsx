"use client";

import { useCallback, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  AreaChart,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Forecast {
  dates: string[];
  p5: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
  outer_base: number[];
  outer_height: number[];
  inner_base: number[];
  inner_height: number[];
  n_simulations: number;
  n_days: number;
  signal_horizon_days: number;
  signal_price: number;
  signal_return_pct: number;
  signal: "UP" | "DOWN";
}

interface PredictionData {
  ticker: string;
  current_price: number;
  predicted_price: number;
  predicted_return_pct: number;
  direction: "UP" | "DOWN";
  signal: "UP" | "DOWN";
  signal_horizon_days: number;
  signal_price: number;
  signal_return_pct: number;
  directional_accuracy: number;
  chart: { dates: string[]; actual: number[]; predicted: number[] };
  backtest: { dates: string[]; strategy: number[]; buy_hold: number[] };
  forecast: Forecast;
  history: { dates: string[]; prices: number[] };
}

const POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY"];

const TARGETS = [
  "Expected return",
  "Up/down probability",
  "Volatility",
  "Relative performance",
  "Distribution of outcomes",
];

const MODEL_STACK = [
  {
    title: "Time-Series",
    detail: "ARIMA, GARCH, regime-switching, Kalman filters for short-horizon structure and volatility.",
  },
  {
    title: "Cross-Sectional Factors",
    detail: "Momentum, value, quality, liquidity, revisions, and sector-relative alpha signals.",
  },
  {
    title: "Machine Learning",
    detail: "Boosted trees and nonlinear tabular models on prices, fundamentals, options, macro, and text.",
  },
  {
    title: "Risk + Execution",
    detail: "Tail-risk estimation, cost-aware portfolio construction, and disciplined execution overlays.",
  },
];

const PIPELINE = [
  "Features",
  "Return/Risk Forecasts",
  "Optimizer",
  "Portfolio",
  "Execution",
];

const TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "rgba(8, 15, 24, 0.96)",
    border: "1px solid rgba(148, 163, 184, 0.16)",
    borderRadius: 16,
    fontSize: 12,
    boxShadow: "0 18px 60px rgba(0, 0, 0, 0.34)",
  },
  labelStyle: { color: "#8a9aab" },
};

// ── Custom tooltip for Monte Carlo chart ──────────────────────────────────────
function ForecastTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = Object.fromEntries(payload.map((p: any) => [p.dataKey, p.value]));

  // Historical point
  if (d.price !== undefined && d.price !== null) {
    return (
      <div style={TOOLTIP_STYLE.contentStyle} className="px-3 py-2">
        <p className="text-slate-400 text-xs mb-1">{label}</p>
        <p className="text-blue-400 font-mono text-sm">
          ${Number(d.price).toFixed(2)}
        </p>
      </div>
    );
  }

  // Forecast point — recover actual percentile values from stacked data
  const p5  = d.outer_base;
  const p95 = p5 !== undefined ? p5 + (d.outer_height ?? 0) : undefined;
  const p25 = d.inner_base;
  const p75 = p25 !== undefined ? p25 + (d.inner_height ?? 0) : undefined;
  const p50 = d.p50;

  if (p50 === undefined) return null;
  return (
    <div style={TOOLTIP_STYLE.contentStyle} className="px-3 py-2 space-y-0.5">
      <p className="text-slate-400 text-xs mb-1">{label}</p>
      {p95 !== undefined && <p className="text-slate-500 font-mono text-xs">p95 ${p95.toFixed(2)}</p>}
      {p75 !== undefined && <p className="text-slate-400 font-mono text-xs">p75 ${p75.toFixed(2)}</p>}
      <p className="text-blue-300 font-mono text-sm font-semibold">p50 ${p50.toFixed(2)}</p>
      {p25 !== undefined && <p className="text-slate-400 font-mono text-xs">p25 ${p25.toFixed(2)}</p>}
      {p5  !== undefined && <p className="text-slate-500 font-mono text-xs">p5  ${p5.toFixed(2)}</p>}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [input, setInput]           = useState("AAPL");
  const [activeTicker, setActive]   = useState("");
  const [data, setData]             = useState<PredictionData | null>(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [horizon, setHorizon]       = useState(60);

  const fetchPrediction = useCallback(async (ticker: string, days = horizon) => {
    setLoading(true);
    setError(null);
    try {
      const res  = await fetch(`/api/predict?ticker=${encodeURIComponent(ticker)}&days=${days}`);
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      setData(json);
      setActive(ticker);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [horizon]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = input.trim().toUpperCase();
    if (t) fetchPrediction(t);
  };

  // ── Chart data helpers ──────────────────────────────────────────────────────
  const priceChartData = data
    ? data.chart.dates.map((d, i) => ({
        date: d.slice(5),
        Actual: data.chart.actual[i],
        Predicted: data.chart.predicted[i],
      }))
    : [];

  const backtestData = data
    ? data.backtest.dates.map((d, i) => ({
        date: d.slice(0, 7),
        Strategy: data.backtest.strategy[i],
        "Buy & Hold": data.backtest.buy_hold[i],
      }))
    : [];

  // Combine historical + bridge + forecast for Monte Carlo chart
  const forecastChartData = data
    ? [
        ...data.history.dates.map((d, i) => ({
          date: d.slice(5),
          price: data.history.prices[i],
          p50: null as number | null,
          outer_base: null as number | null,
          outer_height: null as number | null,
          inner_base: null as number | null,
          inner_height: null as number | null,
        })),
        // Bridge: bands pinch to current price
        {
          date: "→Now",
          price: data.current_price,
          p50: data.current_price,
          outer_base: data.current_price,
          outer_height: 0,
          inner_base: data.current_price,
          inner_height: 0,
        },
        ...data.forecast.dates.map((d, i) => ({
          date: d.slice(5),
          price: null as number | null,
          p50: data.forecast.p50[i],
          outer_base: data.forecast.outer_base[i],
          outer_height: data.forecast.outer_height[i],
          inner_base: data.forecast.inner_base[i],
          inner_height: data.forecast.inner_height[i],
        })),
      ]
    : [];

  // Y-axis domain for forecast chart
  const yMin = data
    ? Math.min(data.current_price, ...data.history.prices, ...data.forecast.p5) * 0.97
    : 0;
  const yMax = data
    ? Math.max(data.current_price, ...data.history.prices, ...data.forecast.p95) * 1.03
    : 100;

  const nextDayUp = data ? data.predicted_price >= data.current_price : false;
  const signalUp = data?.signal === "UP";
  const priceDiff = data ? data.predicted_price - data.current_price : 0;
  const signalHorizonLabel = data ? `${data.signal_horizon_days}d` : "";

  // 30-day forecast stats
  const fc30 = data ? data.forecast.p50[Math.min(29, data.forecast.p50.length - 1)] : null;
  const fc30ret = fc30 && data ? ((fc30 - data.current_price) / data.current_price) * 100 : null;
  const fc30_p5  = data ? data.forecast.p5[Math.min(29, data.forecast.p5.length - 1)] : null;
  const fc30_p95 = data ? data.forecast.p95[Math.min(29, data.forecast.p95.length - 1)] : null;

  return (
    <div className="min-h-screen px-4 py-8 text-white md:px-6 md:py-10">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 grid gap-6 lg:grid-cols-[1.45fr_0.95fr]">
          <section className="rounded-[28px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(108,196,255,0.18),transparent_30%),linear-gradient(180deg,rgba(8,15,24,0.9),rgba(5,11,18,0.82))] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.28)] md:p-8">
            <div className="mb-5 flex flex-wrap items-center gap-3 text-[11px] uppercase tracking-[0.32em] text-slate-400">
              <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-cyan-200">
                Quant Signal Lab
              </span>
              <span>Forecast returns, risk, and distributions</span>
            </div>
            <h1 className="max-w-3xl text-4xl font-semibold leading-tight text-slate-50 md:text-6xl">
              Model the stock like a <span className="text-cyan-300">quant desk</span>, not like a fortune teller.
            </h1>
            <p className="mt-5 max-w-2xl text-sm leading-7 text-slate-300 md:text-base">
              Professional systems usually avoid predicting an exact future price. They forecast expected returns,
              directional probabilities, volatility, and outcome ranges, then layer optimization and execution on top.
            </p>
            <div className="mt-8 grid gap-4 md:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="mb-3 text-xs uppercase tracking-[0.24em] text-slate-500">Targets that actually matter</p>
                <div className="grid gap-2 sm:grid-cols-2">
                  {TARGETS.map((target) => (
                    <div
                      key={target}
                      className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2 text-sm text-slate-200"
                    >
                      {target}
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-2xl border border-amber-300/10 bg-amber-300/[0.05] p-4">
                <p className="mb-3 text-xs uppercase tracking-[0.24em] text-amber-100/70">Pipeline</p>
                <div className="flex flex-wrap gap-2">
                  {PIPELINE.map((step, index) => (
                    <div key={step} className="flex items-center gap-2">
                      <span className="rounded-full border border-amber-200/15 bg-black/20 px-3 py-1 font-mono text-xs text-amber-100">
                        {step}
                      </span>
                      {index < PIPELINE.length - 1 && <span className="text-amber-200/40">→</span>}
                    </div>
                  ))}
                </div>
                <p className="mt-4 text-sm leading-6 text-amber-50/80">
                  The app still computes a day-one estimate and Monte Carlo path fan, but the framing is now alpha,
                  risk, and scenario analysis.
                </p>
              </div>
            </div>
          </section>

          <section className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(10,18,27,0.88),rgba(5,11,18,0.76))] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.24)]">
            <p className="mb-4 text-xs uppercase tracking-[0.24em] text-slate-500">Model stack</p>
            <div className="space-y-3">
              {MODEL_STACK.map((item, index) => (
                <div key={item.title} className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <h2 className="text-base font-semibold text-slate-100">{item.title}</h2>
                    <span className="font-mono text-xs text-slate-500">0{index + 1}</span>
                  </div>
                  <p className="text-sm leading-6 text-slate-400">{item.detail}</p>
                </div>
              ))}
            </div>
          </section>
        </div>

        <section className="mb-8 rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(8,15,24,0.82),rgba(5,11,18,0.7))] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.18)] md:p-6">
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Run a signal</p>
              <p className="mt-1 text-sm text-slate-300">
                Analyze one name, then interpret the output as forecast distribution and tactical signal quality.
              </p>
            </div>
            <div className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-xs text-slate-400">
              Gradient boosting + Monte Carlo
            </div>
          </div>

          <form onSubmit={handleSubmit} className="mb-4 flex flex-wrap items-center gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value.toUpperCase())}
              placeholder="Ticker (e.g. AAPL)"
              className="w-44 rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 font-mono text-sm text-white placeholder-slate-500 outline-none transition focus:border-cyan-400/60"
            />
            <div className="flex flex-wrap items-center gap-2 text-sm text-slate-400">
              <span className="mr-1">Forecast horizon</span>
              {[30, 60, 90].map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => setHorizon(d)}
                  className={`rounded-full px-3 py-2 font-mono text-xs transition ${
                    horizon === d
                      ? "border border-cyan-300/30 bg-cyan-400/15 text-cyan-100"
                      : "border border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-white"
                  }`}
                >
                  {d}d
                </button>
              ))}
            </div>
            <button
              type="submit"
              disabled={loading}
              className="rounded-2xl border border-cyan-300/20 bg-cyan-400/90 px-6 py-3 text-sm font-medium text-slate-950 transition hover:bg-cyan-300 disabled:opacity-50"
            >
              {loading ? "Running…" : "Run Quant Model"}
            </button>
          </form>

          <div className="flex flex-wrap gap-2">
            {POPULAR.map((t) => (
              <button
                key={t}
                onClick={() => { setInput(t); fetchPrediction(t); }}
                className={`rounded-full px-3 py-1.5 font-mono text-xs transition ${
                  activeTicker === t && data
                    ? "border border-cyan-300/30 bg-cyan-400/15 text-cyan-100"
                    : "border border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-white"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="mb-6 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
            {error}
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center gap-4 py-24 text-slate-500">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
            <p className="text-sm">
              Pulling market data, fitting the model, and running{" "}
              <span className="font-mono text-cyan-300">1,000 scenario paths</span> for{" "}
              <span className="font-mono text-cyan-300">{input}</span>.
            </p>
          </div>
        )}

        {data && !loading && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Spot Price" value={`$${data.current_price.toFixed(2)}`} sub={data.ticker} accent="gray" />
              <StatCard
                label="1D Expected Move"
                value={`$${data.predicted_price.toFixed(2)}`}
                sub={`${priceDiff >= 0 ? "+" : ""}$${priceDiff.toFixed(2)} · ${data.predicted_return_pct > 0 ? "+" : ""}${data.predicted_return_pct.toFixed(3)}%`}
                accent={nextDayUp ? "green" : "red"}
              />
              <StatCard
                label="Primary Alpha Signal"
                value={signalUp ? "LONG BIAS" : "RISK OFF"}
                sub={`${signalHorizonLabel} median path · $${data.signal_price.toFixed(2)} · ${data.signal_return_pct > 0 ? "+" : ""}${data.signal_return_pct.toFixed(3)}%`}
                accent={signalUp ? "green" : "red"}
                large
              />
              <StatCard
                label="Directional Hit Rate"
                value={`${(data.directional_accuracy * 100).toFixed(1)}%`}
                sub={`+${((data.directional_accuracy - 0.5) * 100).toFixed(1)}pp vs random`}
                accent="blue"
              />
            </div>

            <div className="grid gap-4 lg:grid-cols-[1.45fr_0.55fr]">
              <ChartCard
                title={`Forecast Distribution Fan — ${data.forecast.n_days}-day horizon`}
                subtitle={`${data.forecast.n_simulations.toLocaleString()} Monte Carlo paths. Median path plus 50% and 90% bands to frame range, not certainty.`}
              >
                {fc30 && fc30_p5 && fc30_p95 && (
                  <div className="mb-5 flex flex-wrap gap-4 text-xs font-mono">
                    <div>
                      <span className="text-slate-500">30d median </span>
                      <span className={`font-semibold ${fc30ret! >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                        ${fc30.toFixed(2)} ({fc30ret! >= 0 ? "+" : ""}{fc30ret!.toFixed(1)}%)
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500">30d 90% band </span>
                      <span className="text-slate-300">${fc30_p5.toFixed(2)} - ${fc30_p95.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">signal horizon </span>
                      <span className="text-slate-300">{signalHorizonLabel}</span>
                    </div>
                  </div>
                )}

                <ResponsiveContainer width="100%" height={320}>
                  <ComposedChart data={forecastChartData}>
                    <defs>
                      <linearGradient id="outerGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#6cc4ff" stopOpacity={0.20} />
                        <stop offset="95%" stopColor="#6cc4ff" stopOpacity={0.05} />
                      </linearGradient>
                      <linearGradient id="innerGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#63e6d8" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#63e6d8" stopOpacity={0.10} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: "#637588", fontSize: 10 }}
                      interval={Math.floor(forecastChartData.length / 7)}
                    />
                    <YAxis
                      domain={[yMin, yMax]}
                      tick={{ fill: "#637588", fontSize: 11 }}
                      tickFormatter={(v) => `$${v.toFixed(0)}`}
                      width={58}
                    />
                    <Tooltip content={<ForecastTooltip />} />

                    <Area type="monotone" dataKey="outer_base" stackId="outer" stroke="none" fill="transparent" legendType="none" connectNulls={false} isAnimationActive={false} />
                    <Area
                      type="monotone"
                      dataKey="outer_height"
                      stackId="outer"
                      stroke="#9ed2ff"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      fill="url(#outerGrad)"
                      name="90% Band"
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                    <Area type="monotone" dataKey="inner_base" stackId="inner" stroke="none" fill="transparent" legendType="none" connectNulls={false} isAnimationActive={false} />
                    <Area
                      type="monotone"
                      dataKey="inner_height"
                      stackId="inner"
                      stroke="#63e6d8"
                      strokeWidth={1}
                      fill="url(#innerGrad)"
                      name="50% Band"
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                    <Line type="monotone" dataKey="p50" stroke="#f6c46d" strokeWidth={2} strokeDasharray="6 3" dot={false} name="Median Path" connectNulls={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="price" stroke="#6cc4ff" strokeWidth={2} dot={false} name="Observed Price" connectNulls={false} isAnimationActive={false} />
                    <ReferenceLine
                      x="→Now"
                      stroke="#475569"
                      strokeDasharray="4 2"
                      label={{ value: "Now", fill: "#637588", fontSize: 10, position: "top" }}
                    />
                    <Legend wrapperStyle={{ color: "#8a9aab", fontSize: 11 }} formatter={(value) => value === "outer_base" || value === "inner_base" ? null : value} />
                  </ComposedChart>
                </ResponsiveContainer>
              </ChartCard>

              <div className="grid gap-4">
                <InfoCard
                  title="Interpretation"
                  body="Think in expected edge per trade, not one-shot certainty. A weak but repeatable forecast can still be useful inside a diversified portfolio."
                />
                <InfoCard
                  title="What This Signal Means"
                  body={`The system is using the model's day-one expectation plus forward scenario paths to express a ${signalUp ? "pro-risk" : "defensive"} stance over the next ${data.signal_horizon_days} trading days.`}
                />
                <InfoCard
                  title="Beginner Mistake To Avoid"
                  body="Obsessing over model class while ignoring target definition, data quality, lookahead bias, costs, and regime robustness."
                />
              </div>
            </div>

            <ChartCard
              title="Residual Fit: Observed vs Model Estimate"
              subtitle="Held-out period comparison. This is still a sanity check, but the dashboard is framed as signal extraction rather than exact point prediction."
            >
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={priceChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: "#637588", fontSize: 11 }} interval={Math.floor(priceChartData.length / 6)} />
                  <YAxis tick={{ fill: "#637588", fontSize: 11 }} domain={["auto", "auto"]} tickFormatter={(v) => `$${v}`} width={58} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: unknown) => [`$${Number(v).toFixed(2)}`]} />
                  <Legend wrapperStyle={{ color: "#8a9aab", fontSize: 11 }} />
                  <Line type="monotone" dataKey="Actual" stroke="#6cc4ff" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="Predicted" stroke="#ff8a7b" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartCard>

            <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
              <ChartCard
                title="Tactical Backtest: Signal vs Passive"
                subtitle="Long when the model tilts positive, flat otherwise. No costs are included, so treat this as directional intuition, not investable PnL."
              >
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={backtestData}>
                    <defs>
                      <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#63e6d8" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#63e6d8" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="date" tick={{ fill: "#637588", fontSize: 11 }} interval={Math.floor(backtestData.length / 5)} />
                    <YAxis tick={{ fill: "#637588", fontSize: 11 }} tickFormatter={(v) => `${v.toFixed(0)}%`} width={48} />
                    <Tooltip {...TOOLTIP_STYLE} formatter={(v: unknown) => [`${Number(v).toFixed(1)}%`]} />
                    <Legend wrapperStyle={{ color: "#8a9aab", fontSize: 11 }} />
                    <Area type="monotone" dataKey="Strategy" stroke="#63e6d8" fill="url(#stratGrad)" strokeWidth={2} />
                    <Line type="monotone" dataKey="Buy & Hold" stroke="#f6c46d" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>

              <section className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(8,15,24,0.82),rgba(5,11,18,0.72))] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.18)]">
                <p className="mb-4 text-xs uppercase tracking-[0.24em] text-slate-500">Practical progression</p>
                <div className="space-y-3">
                  {[
                    "Linear factor model",
                    "Regularized regression",
                    "Gradient boosting",
                    "Regime and risk overlay",
                    "Cost-aware optimizer",
                  ].map((step, index) => (
                    <div key={step} className="flex items-start gap-3 rounded-2xl border border-white/8 bg-white/[0.03] p-4">
                      <span className="mt-0.5 rounded-full border border-cyan-300/20 bg-cyan-400/10 px-2 py-1 font-mono text-xs text-cyan-200">
                        0{index + 1}
                      </span>
                      <div>
                        <p className="text-sm font-semibold text-slate-100">{step}</p>
                        <p className="mt-1 text-sm leading-6 text-slate-400">
                          {index === 0 && "Start with interpretable return drivers and benchmark residual performance."}
                          {index === 1 && "Control overfitting and let the model shrink noisy predictors."}
                          {index === 2 && "Capture nonlinear interactions once feature engineering is stable."}
                          {index === 3 && "Adapt risk budgets when regimes, volatility, or dispersion change."}
                          {index === 4 && "Translate forecasts into portfolio weights that survive costs and turnover."}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>

            <p className="pb-4 text-center text-xs text-slate-600">
              Educational use only. Signals are noisy, costs matter, and scenario fans are not guarantees.
            </p>
          </div>
        )}

        {!data && !loading && !error && (
          <div className="flex flex-col items-center justify-center rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(8,15,24,0.72),rgba(5,11,18,0.62))] py-24 text-slate-500">
            <span className="mb-4 font-mono text-sm uppercase tracking-[0.32em] text-slate-600">No active forecast</span>
            <p className="max-w-md text-center text-sm leading-7 text-slate-400">
              Select a ticker to generate expected move, signal bias, and forward outcome bands instead of a single naive price target.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function InfoCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(8,15,24,0.82),rgba(5,11,18,0.7))] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.18)]">
      <p className="mb-2 text-xs uppercase tracking-[0.24em] text-slate-500">{title}</p>
      <p className="text-sm leading-7 text-slate-300">{body}</p>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StatCard({
  label, value, sub, accent, large,
}: {
  label: string; value: string; sub: string;
  accent: "gray" | "green" | "red" | "blue"; large?: boolean;
}) {
  const colors = { gray: "text-slate-100", green: "text-emerald-300", red: "text-rose-300", blue: "text-cyan-300" };
  return (
    <div className="rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(8,15,24,0.86),rgba(5,11,18,0.74))] p-5 shadow-[0_20px_80px_rgba(0,0,0,0.16)]">
      <p className="mb-2 text-xs font-medium uppercase tracking-[0.22em] text-slate-500">{label}</p>
      <p className={`font-mono font-bold ${large ? "text-3xl" : "text-xl"} ${colors[accent]}`}>{value}</p>
      <p className="mt-1 font-mono text-xs text-slate-600">{sub}</p>
    </div>
  );
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(8,15,24,0.84),rgba(5,11,18,0.72))] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.2)]">
      <h2 className="mb-1 text-sm font-semibold text-slate-100">{title}</h2>
      <p className="mb-5 text-xs leading-6 text-slate-500">{subtitle}</p>
      {children}
    </div>
  );
}
