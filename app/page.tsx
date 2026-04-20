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
}

interface PredictionData {
  ticker: string;
  current_price: number;
  predicted_price: number;
  predicted_return_pct: number;
  direction: "UP" | "DOWN";
  directional_accuracy: number;
  chart: { dates: string[]; actual: number[]; predicted: number[] };
  backtest: { dates: string[]; strategy: number[]; buy_hold: number[] };
  forecast: Forecast;
  history: { dates: string[]; prices: number[] };
}

const POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY"];

const TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#94a3b8" },
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

  const isUp = data?.direction === "UP";
  const priceDiff = data ? data.predicted_price - data.current_price : 0;

  // 30-day forecast stats
  const fc30 = data ? data.forecast.p50[Math.min(29, data.forecast.p50.length - 1)] : null;
  const fc30ret = fc30 && data ? ((fc30 - data.current_price) / data.current_price) * 100 : null;
  const fc30_p5  = data ? data.forecast.p5[Math.min(29, data.forecast.p5.length - 1)] : null;
  const fc30_p95 = data ? data.forecast.p95[Math.min(29, data.forecast.p95.length - 1)] : null;

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-8 text-white">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold tracking-tight mb-0.5">
            Stock Predictor <span className="text-blue-400">ML</span>
          </h1>
          <p className="text-slate-500 text-sm">
            Gradient Boosting · Monte Carlo simulation · next-day + multi-week forecast
          </p>
        </div>

        {/* Search + horizon */}
        <form onSubmit={handleSubmit} className="flex flex-wrap gap-3 mb-4 items-center">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            placeholder="Ticker (e.g. AAPL)"
            className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 w-40 font-mono text-sm"
          />
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span>Forecast horizon:</span>
            {[30, 60, 90].map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setHorizon(d)}
                className={`px-3 py-1.5 rounded-md font-mono text-xs transition-colors ${
                  horizon === d
                    ? "bg-blue-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                }`}
              >
                {d}d
              </button>
            ))}
          </div>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-6 py-2.5 rounded-lg font-medium text-sm transition-colors"
          >
            {loading ? "Analyzing…" : "Predict"}
          </button>
        </form>

        {/* Quick tickers */}
        <div className="flex flex-wrap gap-2 mb-8">
          {POPULAR.map((t) => (
            <button
              key={t}
              onClick={() => { setInput(t); fetchPrediction(t); }}
              className={`px-3 py-1 rounded-md text-xs font-mono transition-colors ${
                activeTicker === t && data
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {error && (
          <div className="bg-red-950/50 border border-red-800 rounded-lg p-4 mb-6 text-red-300 text-sm">
            ⚠ {error}
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-4 text-slate-500">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm">
              Downloading data · training model · running{" "}
              <span className="font-mono text-blue-400">1,000 Monte Carlo paths</span> for{" "}
              <span className="font-mono text-blue-400">{input}</span>…
            </p>
          </div>
        )}

        {data && !loading && (
          <div className="space-y-6">

            {/* ── Row 1: next-day stat cards ── */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Current Price" value={`$${data.current_price.toFixed(2)}`} sub={data.ticker} accent="gray" />
              <StatCard
                label="Predicted Tomorrow"
                value={`$${data.predicted_price.toFixed(2)}`}
                sub={`${priceDiff >= 0 ? "+" : ""}$${priceDiff.toFixed(2)} · ${data.predicted_return_pct > 0 ? "+" : ""}${data.predicted_return_pct.toFixed(3)}%`}
                accent={isUp ? "green" : "red"}
              />
              <StatCard label="Signal" value={isUp ? "▲ BUY" : "▼ SELL"} sub="next trading day" accent={isUp ? "green" : "red"} large />
              <StatCard
                label="Directional Accuracy"
                value={`${(data.directional_accuracy * 100).toFixed(1)}%`}
                sub={`+${((data.directional_accuracy - 0.5) * 100).toFixed(1)}pp vs random`}
                accent="blue"
              />
            </div>

            {/* ── Monte Carlo forecast chart ── */}
            <ChartCard
              title={`Monte Carlo Price Forecast — ${data.forecast.n_days}-day horizon`}
              subtitle={`${data.forecast.n_simulations.toLocaleString()} simulated paths · GBM with ML-informed day-1 drift · shaded bands show 50% and 90% confidence intervals`}
            >
              {/* 30-day forecast mini-stats */}
              {fc30 && fc30_p5 && fc30_p95 && (
                <div className="flex flex-wrap gap-4 mb-5 text-xs font-mono">
                  <div>
                    <span className="text-slate-500">30d median target </span>
                    <span className={`font-semibold ${fc30ret! >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      ${fc30.toFixed(2)} ({fc30ret! >= 0 ? "+" : ""}{fc30ret!.toFixed(1)}%)
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">90% CI at 30d </span>
                    <span className="text-slate-300">${fc30_p5.toFixed(2)} – ${fc30_p95.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">simulations </span>
                    <span className="text-slate-300">{data.forecast.n_simulations.toLocaleString()}</span>
                  </div>
                </div>
              )}

              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={forecastChartData}>
                  <defs>
                    <linearGradient id="outerGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.18} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
                    </linearGradient>
                    <linearGradient id="innerGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.15} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "#475569", fontSize: 10 }}
                    interval={Math.floor(forecastChartData.length / 7)}
                  />
                  <YAxis
                    domain={[yMin, yMax]}
                    tick={{ fill: "#475569", fontSize: 11 }}
                    tickFormatter={(v) => `$${v.toFixed(0)}`}
                    width={58}
                  />
                  <Tooltip content={<ForecastTooltip />} />

                  {/* Outer 90% CI band — stacked: transparent base + colored top */}
                  <Area
                    type="monotone"
                    dataKey="outer_base"
                    stackId="outer"
                    stroke="none"
                    fill="transparent"
                    legendType="none"
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="outer_height"
                    stackId="outer"
                    stroke="#93c5fd"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    fill="url(#outerGrad)"
                    name="90% CI"
                    connectNulls={false}
                    isAnimationActive={false}
                  />

                  {/* Inner 50% CI band — independent stack */}
                  <Area
                    type="monotone"
                    dataKey="inner_base"
                    stackId="inner"
                    stroke="none"
                    fill="transparent"
                    legendType="none"
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="inner_height"
                    stackId="inner"
                    stroke="#60a5fa"
                    strokeWidth={1}
                    fill="url(#innerGrad)"
                    name="50% CI"
                    connectNulls={false}
                    isAnimationActive={false}
                  />

                  {/* Median forecast */}
                  <Line
                    type="monotone"
                    dataKey="p50"
                    stroke="#93c5fd"
                    strokeWidth={2}
                    strokeDasharray="6 3"
                    dot={false}
                    name="Median forecast"
                    connectNulls={false}
                    isAnimationActive={false}
                  />

                  {/* Historical price */}
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#60a5fa"
                    strokeWidth={2}
                    dot={false}
                    name="Historical price"
                    connectNulls={false}
                    isAnimationActive={false}
                  />

                  {/* "Today" marker */}
                  <ReferenceLine
                    x="→Now"
                    stroke="#475569"
                    strokeDasharray="4 2"
                    label={{ value: "Today", fill: "#475569", fontSize: 10, position: "top" }}
                  />

                  <Legend
                    wrapperStyle={{ color: "#94a3b8", fontSize: 11 }}
                    formatter={(value) =>
                      value === "outer_base" || value === "inner_base" ? null : value
                    }
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* ── Price accuracy chart ── */}
            <ChartCard
              title="Model Accuracy: Actual vs Predicted Price"
              subtitle="Last 120 trading days of the test set — how well the model tracked real prices"
            >
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={priceChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: "#475569", fontSize: 11 }} interval={Math.floor(priceChartData.length / 6)} />
                  <YAxis tick={{ fill: "#475569", fontSize: 11 }} domain={["auto", "auto"]} tickFormatter={(v) => `$${v}`} width={58} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: unknown) => [`$${Number(v).toFixed(2)}`]} />
                  <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 11 }} />
                  <Line type="monotone" dataKey="Actual" stroke="#60a5fa" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="Predicted" stroke="#f87171" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* ── Backtest ── */}
            <ChartCard
              title="Backtest: Strategy vs Buy & Hold"
              subtitle="Long when model predicts ↑, flat otherwise · test set only · no transaction costs"
            >
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={backtestData}>
                  <defs>
                    <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#34d399" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: "#475569", fontSize: 11 }} interval={Math.floor(backtestData.length / 5)} />
                  <YAxis tick={{ fill: "#475569", fontSize: 11 }} tickFormatter={(v) => `${v.toFixed(0)}%`} width={48} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: unknown) => [`${Number(v).toFixed(1)}%`]} />
                  <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 11 }} />
                  <Area type="monotone" dataKey="Strategy" stroke="#34d399" fill="url(#stratGrad)" strokeWidth={2} />
                  <Line type="monotone" dataKey="Buy & Hold" stroke="#60a5fa" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>

            <p className="text-xs text-slate-700 text-center pb-4">
              Educational purposes only · not financial advice · Monte Carlo uses Geometric Brownian Motion
            </p>
          </div>
        )}

        {!data && !loading && !error && (
          <div className="flex flex-col items-center justify-center py-24 text-slate-700">
            <span className="text-5xl mb-4">📊</span>
            <p>Select a ticker above to generate a forecast</p>
          </div>
        )}
      </div>
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
  const colors = { gray: "text-slate-100", green: "text-emerald-400", red: "text-red-400", blue: "text-blue-400" };
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <p className="text-slate-500 text-xs font-medium uppercase tracking-widest mb-2">{label}</p>
      <p className={`font-bold font-mono ${large ? "text-3xl" : "text-xl"} ${colors[accent]}`}>{value}</p>
      <p className="text-slate-600 text-xs mt-1 font-mono">{sub}</p>
    </div>
  );
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <h2 className="text-sm font-semibold text-slate-200 mb-0.5">{title}</h2>
      <p className="text-slate-600 text-xs mb-5">{subtitle}</p>
      {children}
    </div>
  );
}
