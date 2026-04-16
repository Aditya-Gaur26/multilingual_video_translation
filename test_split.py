import re
text = "सभी को नमस्कार, संरचना विश्लेषण 1 में आपका स्वागत है। यह पाठ्यक्रम एन. पी. टी. ई. एल. द्वारा ऑनलाइन प्रमाणन पाठ्यक्रम के रूप में और पहल के रूप में पेश किया जा रहा है।"
# Remove dots that are used for acronyms (i.e. single letter + dot)
# But in Hindi it's En. P. T. ... it's multi-char.
# Actually, just replace all English '.' with space in Hindi text, since Hindi uses '।' for periods.
clean_text = text.replace('.', ' ')
print("Cleaned:", clean_text)
