"""Every word the participant sees, in one place.

Kept short and concrete. Keys are written as they appear on the keyboard. The task is described
in terms of what is heard ("the last two beeps start together / out of sync"), never in terms of
the hypothesis.
"""

QUESTIONNAIRE = (
    {"name": "age", "label": "Age (years)", "type": "int", "min": 16, "max": 99},
    {"name": "gender", "label": "Gender", "type": "choice",
     "options": ["Woman", "Man", "Non-binary", "Prefer to self-describe", "Prefer not to say"]},
    {"name": "gender_text", "label": "Self-description (optional)", "type": "text",
     "optional": True},
    {"name": "music_years", "label": "Years of formal music training", "type": "int",
     "min": 0, "max": 70},
    {"name": "plays_now", "label": "Do you currently play an instrument or sing regularly?",
     "type": "choice", "options": ["No", "Yes"]},
    {"name": "hearing_problems", "label": "Do you know of any problems with your hearing?",
     "type": "choice", "options": ["No", "Yes", "Not sure"]},
    {"name": "tinnitus", "label": "Do you have ringing in your ears (tinnitus)?",
     "type": "choice", "options": ["No", "Yes", "Not sure"]},
    {"name": "prior_experiments", "label": "Have you taken part in listening experiments before?",
     "type": "choice", "options": ["No", "Yes"]},
)

DEBRIEF = (
    {"name": "difficulty", "label": "Overall, how hard was the task? (1 very easy, 5 very hard)",
     "type": "choice", "options": ["1", "2", "3", "4", "5"]},
    {"name": "strategy", "label": "How did you decide which sequence ended out of sync?",
     "type": "longtext", "optional": True},
    {"name": "problems", "label": "Anything that disturbed you (sound, headphones, noise)?",
     "type": "longtext", "optional": True},
)

T = {
    "about_title": "A few questions first",
    "about_body": "These help us describe the group of people who took part. "
                  "Press Enter when you are done.",

    "intro1_title": "Welcome",
    "intro1_body": "You will hear short sequences of high and low beeps through the headphones.\n\n"
                   "Each trial has two sequences, one after the other. Your task is about the very "
                   "end of each sequence.",
    "intro2_title": "What to listen for",
    "intro2_body": "Each sequence ends with one high and one low beep. In one of the two sequences "
                   "these last two beeps start together. In the other they are slightly out of "
                   "sync: one starts a little before the other.\n\n"
                   "Pick the sequence whose ending was out of sync. The beeps before the ending "
                   "may or may not be together: ignore them and listen to the very end.",
    "intro3_title": "How to answer",
    "intro3_body": "The left box lights up during the first sequence, the right box during the "
                   "second.\n\n"
                   "Press  ←  (or 1)  if the first sequence ended out of sync.\n"
                   "Press  →  (or 2)  if the second one did.\n\n"
                   "After each answer you will see whether it was right. Many trials are meant to "
                   "be hard: when you are not sure, make your best guess. "
                   "Press P at any time to pause.",

    "check_title": "Sound check",
    "check_body": "You will now hear one trial. Just listen.",
    "check_q_title": "Could you hear the beeps clearly in both ears, at a comfortable level?",
    "check_q_body": "Press Y for yes, N for no.",
    "check_fail_title": "Please tell the experimenter",
    "check_fail_body": "They will adjust the headphones or the level.\n\n"
                       "Experimenter: press Space to repeat the sound check.",

    "demo_title": "Example {i} of {n}",
    "demo_body": "Listen. The labels show which sequence ended out of sync.",
    "demo_again_title": "Hear it again?",
    "demo_again_body": "Press R to replay this example, or Space to continue.",

    "practice_title": "Practice",
    "practice_body": "{n} practice trials, all fairly easy. Answer with ← or →.",
    "practice_pass_title": "Well done",
    "practice_pass_body": "{k} of {n} correct.",
    "practice_retry_title": "{k} of {n} correct",
    "practice_retry_body": "Let's practise once more. Listen to the very last two beeps of each "
                           "sequence: do they start together?",
    "practice_fail_title": "Please tell the experimenter",
    "practice_fail_body": "Experimenter: the practice criterion was not reached.\n"
                          "C = continue anyway,  Q = end the session.",

    "exposure_title": "A few more examples",
    "exposure_body": "In these, the earlier beeps are not always together. The task is the same: "
                     "listen to the very end.",

    "main_title": "The main part",
    "main_body": "There are {n} parts of about 5 minutes each, with a short break between them.\n\n"
                 "The task adapts to you: it gets harder after correct answers and easier after "
                 "mistakes, so many trials will feel difficult. That is expected. Keep guessing "
                 "when you are unsure.",

    "block_title": "Part {i} of {n}",
    "block_body": "Press Space when you are ready.",
    "rest_title": "Part {i} of {n} done",
    "rest_body": "Take a short break. Stretch, close your eyes, have some water.",
    "rest_long_body": "You are halfway. Take a longer break if you like.",

    "resume_title": "Welcome back",
    "resume_body": "We will continue where you left off, after a few warm-up trials.",

    "debrief_title": "Last questions",
    "debrief_body": "Press Enter when you are done.",
    "end_title": "Thank you!",
    "end_body": "That is the end of the experiment. Please let the experimenter know.",

    "prompt_listen": "Listen",
    "prompt_respond": "Which sequence ended out of sync?",
    "prompt_reminder": "Please answer:  ← first   → second",
    "caption_sync": "ended together",
    "caption_async": "ended out of sync",
    "correct": "Correct",
    "wrong": "Wrong",
    "paused_title": "Paused",
    "paused_body": "Press Space to continue.",
    "operator_title": "Experimenter menu",
    "operator_body": "Space = back to the experiment\nQ = save and quit",
    "space_hint": "Press Space to continue",
    "enter_hint": "Press Enter to continue",
}
