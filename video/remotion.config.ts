import { Config } from "@remotion/cli/config"

// Audio-only renders (--codec=wav|mp3|aac, to swap a new soundtrack into a rendered film) take no
// video settings.
const audioOnly = process.argv.some((a) => /^--codec=(wav|mp3|aac)$/.test(a))

Config.setVideoImageFormat("jpeg")
Config.setJpegQuality(92)
if (!audioOnly) {
  Config.setCodec("h264")
  Config.setCrf(18)
}
