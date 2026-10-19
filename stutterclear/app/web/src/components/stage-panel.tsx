import { useState } from "react"
import { Volume2 } from "lucide-react"
import { toast } from "sonner"

import { IsoBadge, IsoStage } from "@/components/iso-icon"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Textarea } from "@/components/ui/textarea"
import { sayIt } from "@/lib/api"

export function StagePanel() {
  const [script, setScript] = useState("")
  const [picked, setPicked] = useState("")
  const [busy, setBusy] = useState(false)
  const [label, setLabel] = useState("")

  const speak = async () => {
    const text = picked || window.getSelection()?.toString() || ""
    if (!text.trim()) {
      toast.message("Select a word or a short phrase in the script first.")
      return
    }
    setBusy(true)
    try {
      const r = await sayIt(text.trim().slice(0, 200))
      setLabel(r.label)
      const url = URL.createObjectURL(r.wav)
      const audio = new Audio(url)
      audio.onended = () => URL.revokeObjectURL(url)
      await audio.play()
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card className="glass border-0 bg-transparent ring-0">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <IsoBadge><IsoStage /></IsoBadge>
          On stage
        </CardTitle>
        <CardDescription>Paste the lines. Select a word, then Say it speaks that word in a synthetic voice, only when you click.</CardDescription>
      </CardHeader>
      <CardContent>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="script">Script</FieldLabel>
            <Textarea
              id="script"
              value={script}
              onChange={(e) => setScript(e.target.value)}
              onMouseUp={() => setPicked(window.getSelection()?.toString() ?? "")}
              rows={8}
              placeholder="The lines you are about to say"
            />
            <FieldDescription>{picked.trim() ? `Selected: ${picked.trim()}` : "Nothing selected yet."}</FieldDescription>
          </Field>
          <Button type="button" disabled={busy} onClick={speak}>
            <Volume2 data-icon="inline-start" />
            {busy ? "Speaking…" : "Say it"}
          </Button>
          {label && (
            <Alert>
              <AlertTitle>{label}</AlertTitle>
              <AlertDescription>This voice is synthetic. It is not a recording of a person.</AlertDescription>
            </Alert>
          )}
        </FieldGroup>
      </CardContent>
    </Card>
  )
}
