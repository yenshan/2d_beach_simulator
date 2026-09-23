import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "SHORE 512 — 小さな海の、尽きない表情。",
  description: "横512×縦256ピクセルのビーチ断面。波の高さや周期を変えながら、砕ける波、白い泡、濡れた砂浜を眺めるシミュレーター。",
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ja"><body>{children}</body></html>;
}
