"use client";

import { useCallback, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────
interface PredictionData {
  ticker: string;
  current_price: number;
  predicted_price: number;
  predicted_return_pct: number;
  direction: "UP" | "DOWN";
  directional_accuracy: number;
  chart: { dates: string[]; actual: number[]; predicted: number[] };
  backtest: { dates: string[]; strategy: number[]; buy_hold: number[] };
}

const POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY"];

// ── Tooltip styles ────────────────────────────────────────────────────────────
const tooltipStyle = {
  contentStyle: {
    backgroundColor: "#111827",
    border: "1px solid #374151",
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#e5e7eb" },
};

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [input, setInput] = useState("AAPL");
  const [activeTicker, setActiveTicker] = useState("");
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPrediction = useCallback(async (ticker: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/predict?ticker=${encodeURIComponent(ticker)}`);
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      setData(json);
      setActiveTicker(ticker);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = input.trim().toUpperCase();
    if (t) fetchPrediction(t);
  };

  // Transform API data for charts
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

  const isUp = data?.direction === "UP";
  const priceDiff = data ? data.predicted_price - data.current_price : 0;

  return (
    <div className="min-h-screen bg-gray-950 px-4 py-8">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-2xl">📈</span>
            <h1 className="text-2xl font-bold tracking-tight">
              Stock Predictor <span className="text-blue-400">ML</span>
            </h1>
          </div>
          <p className="text-gray-500 text-sm pl-10">
            Gradient Boosting · 50+ technical indicators · next-day price forecast
          </p>
        </div>

        {/* Search bar */}
        <form onSubmit={handleSubmit} className="flex gap-3 mb-4">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            placeholder="Ticker (e.g. AAPL)"
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 w-44 font-mono text-sm"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-2.5 rounded-lg font-medium text-sm transition-colors"
          >
            {loading ? "Analyzing…" : "Predict"}
          </button>
        </form>

        {/* Quick-pick tickers */}
        <div className="flex flex-wrap gap-2 mb-8">
          {POPULAR.map((t) => (
            <button
              key={t}
              onClick={() => { setInput(t); fetchPrediction(t); }}
              className={`px-3 py-1 rounded-md text-xs font-mono transition-colors ${
                activeTicker === t && data
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-950/50 border border-red-800 rounded-lg p-4 mb-6 text-red-300 text-sm">
            ⚠ {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-4 text-gray-500">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm">Training model for <span className="font-mono text-blue-400">{input}</span>…</p>
            <p className="text-xs text-gray-600">Downloading 4 years of data · computing indicators · fitting GBM</p>
          </div>
        )}

        {/* Dashboard */}
        {data && !loading && (
          <div className="space-y-6">

            {/* Stat cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                label="Current Price"
                value={`$${data.current_price.toFixed(2)}`}
                sub={data.ticker}
                accent="gray"
              />
              <StatCard
                label="Predicted Tomorrow"
                value={`$${data.predicted_price.toFixed(2)}`}
                sub={`${priceDiff >= 0 ? "+" : ""}$${priceDiff.toFixed(2)} (${data.predicted_return_pct > 0 ? "+" : ""}${data.predicted_return_pct.toFixed(3)}%)`}
                accent={isUp ? "green" : "red"}
              />
              <StatCard
                label="Signal"
                value={isUp ? "▲ BUY" : "▼ SELL"}
                sub="next trading day"
                accent={isUp ? "green" : "red"}
                large
              />
              <StatCard
                label="Directional Accuracy"
                value={`${(data.directional_accuracy * 100).toFixed(1)}%`}
                sub={`+${((data.directional_accuracy - 0.5) * 100).toFixed(1)}pp vs random`}
                accent="blue"
              />
            </div>

            {/* Price chart */}
            <ChartCard title="Actual vs Predicted Price" subtitle="Last 120 trading days">
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={priceChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "#6b7280", fontSize: 11 }}
                    interval={Math.floor(priceChartData.length / 6)}
                  />
                  <YAxis
                    tick={{ fill: "#6b7280", fontSize: 11 }}
                    domain={["auto", "auto"]}
                    tickFormatter={(v) => `$${v}`}
                    width={60}
                  />
                  <Tooltip
                    {...tooltipStyle}
                    formatter={(v: unknown) => [`$${Number(v).toFixed(2)}`]}
                  />
                  <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="Actual"
                    stroke="#60a5fa"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="Predicted"
                    stroke="#f87171"
                    dot={false}
                    strokeWidth={1.5}
                    strokeDasharray="5 3"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Backtest chart */}
            <ChartCard
              title="Backtest: Strategy vs Buy & Hold"
              subtitle="Long when model predicts ↑, flat otherwise · test set only · no transaction costs"
            >
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={backtestData}>
                  <defs>
                    <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#34d399" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "#6b7280", fontSize: 11 }}
                    interval={Math.floor(backtestData.length / 5)}
                  />
                  <YAxis
                    tick={{ fill: "#6b7280", fontSize: 11 }}
                    tickFormatter={(v) => `${v.toFixed(0)}%`}
                    width={55}
                  />
                  <Tooltip
                    {...tooltipStyle}
                    formatter={(v: unknown) => [`${Number(v).toFixed(1)}%`]}
                  />
                  <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 12 }} />
                  <Area
                    type="monotone"
                    dataKey="Strategy"
                    stroke="#34d399"
                    fill="url(#stratGrad)"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="Buy & Hold"
                    stroke="#60a5fa"
                    dot={false}
                    strokeWidth={1.5}
                    strokeDasharray="5 3"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Footer note */}
            <p className="text-xs text-gray-700 text-center pb-4">
              Predictions are for educational purposes only. Not financial advice.
            </p>
          </div>
        )}

        {/* Empty state */}
        {!data && !loading && !error && (
          <div className="flex flex-col items-center justify-center py-24 text-gray-700">
            <span className="text-5xl mb-4">📊</span>
            <p>Select a ticker above to generate a prediction</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StatCard({
  label,
  value,
  sub,
  accent,
  large,
}: {
  label: string;
  value: string;
  sub: string;
  accent: "gray" | "green" | "red" | "blue";
  large?: boolean;
}) {
  const colors = {
    gray: "text-gray-100",
    green: "text-emerald-400",
    red: "text-red-400",
    blue: "text-blue-400",
  };
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <p className="text-gray-500 text-xs font-medium uppercase tracking-widest mb-2">
        {label}
      </p>
      <p className={`font-bold font-mono ${large ? "text-3xl" : "text-xl"} ${colors[accent]}`}>
        {value}
      </p>
      <p className="text-gray-600 text-xs mt-1 font-mono">{sub}</p>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h2 className="text-sm font-semibold text-gray-200 mb-0.5">{title}</h2>
      <p className="text-gray-600 text-xs mb-5">{subtitle}</p>
      {children}
    </div>
  );
}
