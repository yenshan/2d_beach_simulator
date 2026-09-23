"use client";

import { useEffect, useRef, useState } from "react";
import { Ocean, type OceanSettings } from "../lib/ocean";
import type { GpuOcean } from "../lib/gpu-ocean";

const presets = [
  { name: "凪", en: "CALM", height: 0.35, period: 8, wind: 0.1 },
  { name: "さざ波", en: "GENTLE", height: 0.8, period: 6, wind: 0.25 },
  { name: "うねり", en: "SWELL", height: 1.7, period: 7, wind: 0.4 },
  { name: "荒波", en: "ROUGH", height: 2.7, period: 4.5, wind: 0.8 },
];
export default function Home() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const ocean = useRef<Ocean | GpuOcean | null>(null);
  const [settings, setSettings] = useState<OceanSettings>({ height: 2.2, period: 6, wind: 0.3, speed: 1 });
  const [playing, setPlaying] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [fps, setFps] = useState<number | null>(null);
  const [mode, setMode] = useState<"gpu" | "simple">("gpu");
  const [attempt, setAttempt] = useState(0);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [particleCount, setParticleCount] = useState(0);
  const [active, setActive] = useState("custom");
  const current = useRef({ settings, playing });
  useEffect(() => { current.current = { settings, playing }; }, [settings, playing]);
  useEffect(() => {
    let stopped = false, frame = 0, previous = 0, lastLabel = 0;
    let simulation: Ocean | GpuOcean | null = null;
    let sampleStart = 0, sampleFrames = 0;
    const resetFps = () => { sampleStart = 0; sampleFrames = 0; previous = 0; setFps(null); };
    document.addEventListener("visibilitychange", resetFps);
    setStatus("loading"); setError(""); setElapsed(0); setFps(null);
    const fail = (message: string) => {
      if (stopped) return;
      cancelAnimationFrame(frame); setFps(null); setError(message); setStatus("error");
    };
    async function start() {
      try {
        if (mode === "gpu") {
          const { GpuOcean } = await import("../lib/gpu-ocean");
          if (stopped) return;
          simulation = await GpuOcean.create(canvas.current!, fail);
          if (stopped) { simulation.dispose(); return; }
          setParticleCount(simulation.count);
        } else {
          const ctx = canvas.current!.getContext("2d", { alpha: false });
          if (!ctx) throw new Error("描画を初期化できませんでした。");
          simulation = new Ocean(ctx);
        }
        ocean.current = simulation;
        simulation.render(0, current.current.settings); setStatus("ready");
        const tick = (now: number) => {
          if (stopped || !simulation) return;
          try {
            const dt = previous ? Math.min((now - previous) / 1000, 0.05) : 0;
            previous = now;
            if (current.current.playing && !document.hidden) {
              simulation.render(dt, current.current.settings);
              if (!sampleStart) {
                sampleStart = now;
              } else {
                sampleFrames++;
                const duration = now - sampleStart;
                if (duration >= 1000) {
                  setFps(sampleFrames * 1000 / duration);
                  sampleStart = now; sampleFrames = 0;
                }
              }
            } else {
              sampleStart = 0; sampleFrames = 0; setFps(null);
            }
            if (now - lastLabel > 500) { setElapsed(simulation.time); lastLabel = now; }
            frame = requestAnimationFrame(tick);
          } catch (cause) {
            console.error(cause); fail("描画を続けられませんでした。再試行してください。");
          }
        };
        frame = requestAnimationFrame(tick);
      } catch (cause) {
        console.error(cause);
        fail(cause instanceof Error && /WebGPU|ブラウザ|簡易/.test(cause.message) ? cause.message : "水の物理演算を初期化できませんでした。再試行するか、簡易表示に切り替えてください。");
      }
    }
    void start();
    return () => {
      stopped = true; cancelAnimationFrame(frame);
      document.removeEventListener("visibilitychange", resetFps);
      if (simulation && "dispose" in simulation) simulation.dispose();
      ocean.current = null;
    };
  }, [mode, attempt]);
  function change(key: keyof OceanSettings, value: number) { setSettings(s => ({ ...s, [key]: value })); setActive("custom"); }
  function reset() { ocean.current?.reset(); ocean.current?.render(0, settings); setElapsed(0); }
  function save() {
    const link = document.createElement("a"); link.download = "shore-512.png";
    const simulation = ocean.current;
    if (!simulation) return;
    link.href = "snapshot" in simulation ? simulation.snapshot() : canvas.current!.toDataURL("image/png"); link.click();
  }
  return <main>
    <header><a className="brand" href="/" aria-label="SHORE ホーム"><span className="brand-icon">≈</span> SHORE<span className="brand-number">512</span></a><span className="header-note">A LITTLE OCEAN, ALWAYS IN MOTION</span><span className="live"><i /> LIVE SIMULATION</span></header>
    <section className="intro"><div className="eyebrow">PIXEL OCEAN STUDY — 001</div><h1>小さな海の、<br className="mobile" />尽きない表情。</h1><p>512 × 256 の海の断面。波をつくって、ただ眺める。</p></section>
    <section className="workspace">
      <div className="ocean-panel"><div className="canvas-top"><span><i className={playing ? "dot" : "dot paused"} /> {status === "loading" ? "水の計算を準備中…" : status === "error" ? "描画を停止中" : playing ? (mode === "gpu" ? "水の物理演算中" : "簡易表示中") : "一時停止中"}</span><span>SIDE VIEW · 512 × 256</span></div>
        <div className="canvas-wrap"><canvas key={`${mode}-${attempt}`} ref={canvas} width={512} height={256} aria-label="左に砂浜、右に沖。右から左へ波が押し寄せる、横512×縦256ピクセルの海の断面アニメーション" />{status !== "ready" && <div className="gpu-message" role="status"><strong>{status === "loading" ? "水の物理演算を準備しています" : "シミュレーションを開始できません"}</strong><p>{status === "loading" ? "初回は少し時間がかかります。" : error}</p>{status === "error" && <div><button onClick={() => setAttempt(v => v + 1)}>再試行</button><button onClick={() => { setMode("simple"); setAttempt(v => v + 1); }}>簡易表示で開く</button></div>}</div>}<span className="compass">← SHORE · SEA</span><span className="scale">─────<br />COAST CROSS SECTION</span></div>
        <div className="transport"><button className="play" disabled={status !== "ready"} onClick={() => setPlaying(p => !p)} aria-label={playing ? "一時停止" : "再生"}>{playing ? "Ⅱ" : "▶"}</button><button className="text-button" disabled={status !== "ready"} onClick={reset}>↺ <span>リセット</span></button><span className="timer">{Math.floor(elapsed / 60).toString().padStart(2, "0")}:{Math.floor(elapsed % 60).toString().padStart(2, "0")}</span><span className="fps" title="直近約1秒の描画ループの平均フレームレート" aria-label="描画フレームレート">{status === "ready" && playing && fps !== null ? fps.toFixed(1) : "—"} <small>FPS</small></span><button className="capture" disabled={status !== "ready"} onClick={save}>↓ <span>画像を保存</span></button></div>
      </div>
      <aside><div className="control-title"><div className="eyebrow">MAKE YOUR WAVES</div><h2>海のコンディション</h2><p>今日の海は、どんな気分？</p></div><div className="presets">{presets.map(p => <button key={p.name} className={active === p.name ? "preset selected" : "preset"} onClick={() => { setSettings(s => ({ ...s, height: p.height, period: p.period, wind: p.wind })); setActive(p.name); }}><span className="wave-symbol">{p.name === "凪" ? "﹏" : p.name === "さざ波" ? "∿" : p.name === "うねり" ? "≈" : "≋"}</span><strong>{p.name}</strong><small>{p.en}</small></button>)}</div>
        <div className="sliders">
          <label><span>波の高さ <output>{settings.height.toFixed(1)}<small> m</small></output></span><input aria-label="波の高さ" type="range" min="0.2" max="3" step="0.1" value={settings.height} onChange={e => change("height", +e.target.value)} /><span className="range-label"><small>低い</small><small>高い</small></span></label>
          <label><span>波の周期 <output>{settings.period.toFixed(1)}<small> s</small></output></span><input aria-label="波の周期" type="range" min="3" max="10" step="0.5" value={settings.period} onChange={e => change("period", +e.target.value)} /><span className="range-label"><small>短い</small><small>長い</small></span></label>
          <label><span>風の強さ <output>{Math.round(settings.wind * 100)}<small> %</small></output></span><input aria-label="風の強さ" type="range" min="0" max="1" step="0.05" value={settings.wind} onChange={e => change("wind", +e.target.value)} /><span className="range-label"><small>穏やか</small><small>強い</small></span></label>
        </div><div className="speed"><span>時間の速さ</span><div>{[0.5, 1, 2].map(s => <button key={s} className={settings.speed === s ? "selected" : ""} onClick={() => setSettings(v => ({ ...v, speed: s }))}>{s}×</button>)}</div></div>
        <div className="observation"><span>↳ 波打ち際を、よく見ると。</span><p>右から生まれた波が、浅い海底で形を変える。打ち寄せた水は重力で沖へ戻り、次の波とぶつかります。最初の波が届くまで、少し眺めてみてください。</p></div>
      </aside>
    </section><footer><span>動きを感じる、512 × 256 ピクセル。</span><span>{mode === "gpu" ? `2D粒子流体 · ${particleCount.toLocaleString()} 粒子 · WebGPU` : "簡易表示 · 手続き的な波"} · 波高は入力強度の目安{mode === "simple" && <button className="text-button" onClick={() => { setMode("gpu"); setAttempt(v => v + 1); }}> GPU版を再試行</button>}</span></footer>
  </main>;
}
