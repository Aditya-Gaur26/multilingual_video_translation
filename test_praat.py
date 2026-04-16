import parselmouth
from parselmouth.praat import call
import sys

source = sys.argv[1]
target = sys.argv[2]
out    = sys.argv[3]

print(f"Loading '{source}' and '{target}'...")
s1 = parselmouth.Sound(source)
s2 = parselmouth.Sound(target)

pitch = s1.to_pitch()
pitch_tier = call(pitch, "Down to PitchTier")

manip = call(s2, "To Manipulation", 0.01, 75, 600)
call([pitch_tier, manip], "Replace pitch tier")

resynth = call(manip, "Get resynthesis (overlap-add)")
resynth.save(out, "WAV")
print(f"Saved {out}")
