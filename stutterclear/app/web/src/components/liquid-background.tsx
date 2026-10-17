import { useEffect, useRef, type RefObject } from "react"
import type { Signals } from "@/hooks/use-signals"

// Full-screen liquid gradient: domain-warped simplex noise. The surface bulges under the cursor
// (more when it moves fast), every click sends a ripple, and the flow speeds up and brightens
// with your voice. Rendered at half resolution: it is soft by design, and that keeps it cheap.

const VERT = `#version 300 es
in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`

const FRAG = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform vec2 uMouse;
uniform float uVel;
uniform float uHover;
uniform float uLevel;
uniform float uLive;
uniform float uCut;
uniform float uBlock;
uniform vec3 uRipples[4];
out vec4 outColor;

vec3 permute(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m;
  m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float fbm(vec2 p) {
  float s = 0.0, a = 0.5;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 3; i++) {
    s += a * snoise(p);
    p = r * p * 2.03 + 7.3;
    a *= 0.5;
  }
  return s;
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  float asp = uRes.x / uRes.y;
  vec2 p = (uv - 0.5) * vec2(asp, 1.0);
  vec2 m = (uMouse - 0.5) * vec2(asp, 1.0);

  // Lens under the cursor: pull the field toward it, harder while the cursor is moving.
  vec2 dm = p - m;
  float pull = exp(-dot(dm, dm) * 6.0) * uHover;
  p -= dm * pull * (0.28 + 0.9 * uVel);

  // Click ripples: a ring that travels outward and fades.
  for (int i = 0; i < 4; i++) {
    vec3 rp = uRipples[i];
    float age = uTime - rp.z;
    if (age < 0.0 || age > 3.0) continue;
    vec2 d = p - (rp.xy - 0.5) * vec2(asp, 1.0);
    float r = length(d);
    float front = age * 0.6;
    float w = exp(-pow((r - front) * 8.0, 2.0)) * exp(-age * 1.3);
    p += d / max(r, 1e-3) * sin((r - front) * 36.0) * w * 0.04;
  }

  // Large, slow shapes: low frequency and a gentle warp read as liquid, not marble.
  float t = uTime * (0.035 + 0.07 * uLevel + 0.02 * uLive);
  vec2 q = vec2(fbm(p * 0.55 + vec2(0.0, t)), fbm(p * 0.55 + vec2(5.2, -0.8 * t)));
  vec2 r = vec2(fbm(p * 0.7 + 1.3 * q + vec2(1.7, 9.2) + 0.6 * t),
                fbm(p * 0.7 + 1.3 * q + vec2(8.3, 2.8) - 0.5 * t));
  float f = fbm(p * 0.6 + 1.5 * r);

  vec3 base   = vec3(0.016, 0.020, 0.045);
  vec3 indigo = vec3(0.15, 0.13, 0.46);
  vec3 blue   = vec3(0.07, 0.28, 0.62);
  vec3 teal   = vec3(0.03, 0.45, 0.48);
  vec3 violet = vec3(0.45, 0.18, 0.66);
  vec3 rose   = vec3(0.85, 0.20, 0.36);

  vec3 col = base;
  col = mix(col, indigo, smoothstep(-0.45, 0.55, f) * 0.6);
  col = mix(col, blue, smoothstep(0.0, 0.7, q.x) * 0.45);
  col = mix(col, teal, smoothstep(0.05, 0.7, r.x) * (0.25 + 0.3 * uLive));
  col = mix(col, violet, smoothstep(0.15, 0.8, r.y) * (0.3 + 0.4 * uBlock));
  col = mix(col, rose, uCut * smoothstep(-0.2, 0.6, f) * 0.45);

  // Soft light under the cursor; the voice lifts the whole field a little.
  col += vec3(0.22, 0.30, 0.62) * pull * 0.2;
  col *= 0.82 + 0.4 * uLevel;

  float v = smoothstep(1.3, 0.2, length((uv - 0.5) * vec2(asp * 0.75, 1.0)));
  col *= mix(0.3, 1.0, v);

  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  col += (n - 0.5) / 255.0; // dither: no banding in the dark gradients
  outColor = vec4(col, 1.0);
}`

const SCALE = 0.5

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!
  gl.shaderSource(s, src)
  gl.compileShader(s)
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader")
  return s
}

export function LiquidBackground({ signals }: { signals: RefObject<Signals> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "low-power" })
    if (!gl) return // the page keeps its plain dark background

    const prog = gl.createProgram()!
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT))
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(prog)
    gl.useProgram(prog)
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(prog, "p")
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
    const u = (name: string) => gl.getUniformLocation(prog, name)
    const U = {
      res: u("uRes"), time: u("uTime"), mouse: u("uMouse"), vel: u("uVel"), hover: u("uHover"),
      level: u("uLevel"), live: u("uLive"), cut: u("uCut"), block: u("uBlock"), ripples: u("uRipples"),
    }

    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches
    const target = { x: 0.5, y: 0.6 }
    const mouse = { x: 0.5, y: 0.6 }
    let hover = 0
    let hoverTarget = 0
    const ripples = new Float32Array(12).fill(-10)
    let rippleIdx = 0
    const t0 = performance.now()

    const onMove = (e: PointerEvent) => {
      target.x = e.clientX / innerWidth
      target.y = 1 - e.clientY / innerHeight
      hoverTarget = 1
    }
    const onLeave = () => (hoverTarget = 0)
    const onDown = (e: PointerEvent) => {
      ripples.set([e.clientX / innerWidth, 1 - e.clientY / innerHeight, (performance.now() - t0) / 1000], rippleIdx * 3)
      rippleIdx = (rippleIdx + 1) % 4
    }
    addEventListener("pointermove", onMove)
    addEventListener("pointerdown", onDown)
    document.documentElement.addEventListener("pointerleave", onLeave)

    const resize = () => {
      canvas.width = Math.max(1, Math.round(innerWidth * SCALE))
      canvas.height = Math.max(1, Math.round(innerHeight * SCALE))
      gl.viewport(0, 0, canvas.width, canvas.height)
    }
    resize()
    addEventListener("resize", resize)

    let raf = 0
    let last = performance.now()
    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      // The bulge glides after the cursor; the gap between them is how fast it is moving.
      const k = 1 - Math.exp(-dt * 5)
      mouse.x += (target.x - mouse.x) * k
      mouse.y += (target.y - mouse.y) * k
      hover += (hoverTarget - hover) * (1 - Math.exp(-dt * 3))
      const vel = Math.min(1, Math.hypot(target.x - mouse.x, target.y - mouse.y) * 5)
      const s = signals.current
      gl.uniform2f(U.res, canvas.width, canvas.height)
      gl.uniform1f(U.time, ((now - t0) / 1000) * (reduced ? 0.3 : 1))
      gl.uniform2f(U.mouse, mouse.x, mouse.y)
      gl.uniform1f(U.vel, reduced ? 0 : vel)
      gl.uniform1f(U.hover, reduced ? 0 : hover)
      gl.uniform1f(U.level, s.level)
      gl.uniform1f(U.live, s.live)
      gl.uniform1f(U.cut, s.cut)
      gl.uniform1f(U.block, s.block)
      gl.uniform3fv(U.ripples, ripples)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      removeEventListener("pointermove", onMove)
      removeEventListener("pointerdown", onDown)
      removeEventListener("resize", resize)
      document.documentElement.removeEventListener("pointerleave", onLeave)
    }
  }, [signals])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10 h-full w-full"
    />
  )
}
