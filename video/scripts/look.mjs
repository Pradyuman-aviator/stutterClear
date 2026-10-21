// Stills of one scene, for checking it by eye: a contact sheet of nine moments and any frames asked for.
//   node scripts/look.mjs <scene-id> [frame ...] [--sheet=f1,f2,...] [--no-sheet]
// Writes out/look/<scene-id>/sheet.png and f<frame>.png and prints their paths. Only three renders run
// at once on this machine (the rest wait for a slot), so several can be asked for together.
import { bundle } from "@remotion/bundler"
import { openBrowser, renderStill, selectComposition } from "@remotion/renderer"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const SLOTS = 3
const LOCKS = path.join(ROOT, ".locks")

async function acquire() {
  fs.mkdirSync(LOCKS, { recursive: true })
  for (;;) {
    for (let i = 0; i < SLOTS; i++) {
      const dir = path.join(LOCKS, `slot-${i}`)
      try {
        fs.mkdirSync(dir)
        fs.writeFileSync(path.join(dir, "pid"), String(process.pid))
        return dir
      } catch {
        try {
          const age = Date.now() - fs.statSync(dir).mtimeMs
          const pid = Number(fs.readFileSync(path.join(dir, "pid"), "utf8"))
          let alive = true
          try {
            process.kill(pid, 0)
          } catch {
            alive = false
          }
          if (!alive || age > 15 * 60 * 1000) fs.rmSync(dir, { recursive: true, force: true })
        } catch {
          // being created right now, or already gone
        }
      }
    }
    await new Promise((r) => setTimeout(r, 1500))
  }
}

const [id, ...rest] = process.argv.slice(2)
if (!id) {
  console.error("usage: node scripts/look.mjs <scene-id> [frame ...] [--sheet=f1,f2,...] [--no-sheet]")
  process.exit(2)
}
const entry = path.join(ROOT, "src", "entries", `${id}.ts`)
if (!fs.existsSync(entry)) {
  console.error(`No scene "${id}" (no ${entry}). Run node scripts/scaffold.mjs?`)
  process.exit(2)
}
const frames = rest.filter((a) => /^\d+$/.test(a)).map(Number)
const sheetArg = rest.find((a) => a.startsWith("--sheet="))
const sheetFrames = sheetArg ? sheetArg.slice(8).split(",").filter(Boolean).map(Number) : []
const sheet = !rest.includes("--no-sheet")
const out = path.join(ROOT, "out", "look", id)
fs.mkdirSync(out, { recursive: true })

const slot = await acquire()
const release = () => fs.rmSync(slot, { recursive: true, force: true })
process.on("SIGINT", () => {
  release()
  process.exit(130)
})
const t0 = Date.now()
try {
  const serveUrl = await bundle({
    entryPoint: entry,
    rootDir: ROOT,
    publicDir: path.join(ROOT, "public"),
    outDir: path.join(ROOT, ".bundles", id),
    enableCaching: true,
    onProgress: () => {},
  })
  const browser = await openBrowser("chrome")
  try {
    const scene = await selectComposition({ serveUrl, id: `Scene-${id}`, inputProps: {}, puppeteerInstance: browser })
    console.log(`scene ${id}: ${scene.durationInFrames} frames (${(scene.durationInFrames / 30).toFixed(1)} s)`)
    if (sheet) {
      const props = { frames: sheetFrames }
      const comp = await selectComposition({ serveUrl, id: `Sheet-${id}`, inputProps: props, puppeteerInstance: browser })
      const file = path.join(out, "sheet.png")
      await renderStill({ composition: comp, serveUrl, output: file, frame: 0, inputProps: props, puppeteerInstance: browser, overwrite: true })
      console.log(file)
    }
    for (const fr of frames) {
      const frame = Math.min(fr, scene.durationInFrames - 1)
      const file = path.join(out, `f${frame}.png`)
      await renderStill({ composition: scene, serveUrl, output: file, frame, inputProps: {}, puppeteerInstance: browser, overwrite: true })
      console.log(file)
    }
  } finally {
    await browser.close({ silent: true })
  }
  console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s`)
} catch (e) {
  console.error(e?.stack ?? String(e))
  process.exitCode = 1
} finally {
  release()
}
