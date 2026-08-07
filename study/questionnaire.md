# NPC Interrogation Study — Pilot Questionnaire (N=10)

The instrument is split into **three parts**, one per moment in the session, so
that a participant never reads a later question before it applies — in
particular, the Part 3 blind check ("one of these was AI") must not be visible
while they are still forming their Part 2 impressions.

All Likert items are **Linear scale 1–7**, labels: 1 = "Strongly disagree",
7 = "Strongly agree".

Tip: instead of entering these by hand, run `createAllParts()` from
`create_google_form.gs` (same folder) in Google Apps Script — it builds all
three forms automatically and prints their live URLs. Paste those URLs into
`form_links.json`, run `python study/build_study_page.py`, and the public page
at `docs/study/` picks them up.

Protocol (facilitator-run, like Ploug et al. 2025): **Part 1** before play →
play first interrogation and submit its verdict → **Part 2** → play second
interrogation and submit its verdict → **Part 3** → debrief (reveal which
detective was which).

Facilitator rules:

1. **Sequential interrogations.** The app lets participants switch detectives
   per question — instruct them to fully interrogate one detective and submit
   its verdict before starting the other, so the Part 2 / Part 3 blocks stay
   clean.
2. **Counterbalance order to 5/5.** The label↔condition mapping is randomized
   once per session (`new_label_mapping()` in ui.py). Peek at researcher view
   (off the participant's screen) and tell them which detective to start with,
   so 5 participants meet the static suspect first and 5 the dynamic one.
3. **Same Participant ID in all three parts.** The three forms write to three
   separate response sheets; the Participant ID is the only thing joining them.
   Fill it in yourself rather than letting the participant retype it.
4. **Linkage to the logs.** Note each participant's session log filename next
   to their Participant ID. Every logged turn and verdict carries both
   `condition` and `label_shown`, so the B0 answer plus the log resolves which
   block was which condition.

---

## Part 1 — Before you play

> **When:** before the participant starts the game.

> This study compares two versions of an interrogation game. Your gameplay is
> logged anonymously (transcript, timings, verdicts). You may stop at any time
> and ask for your data to be deleted. Some details of the study design will
> be explained only after the session.

### Consent and ID

- **0.1** I have read the description above and consent to take part. — [Multiple choice, required] `I consent`
- **PID** Participant ID (filled in by the researcher, e.g. P01). — [Short answer, required]

### Background

- **A1** Age. — [Short answer]
- **A2** Gender (optional). — [Short answer]
- **A3** On average, how many hours per week do you play video games? — [Short answer, required]
- **A4** I am familiar with role-playing / detective games. — [Linear scale 1–7]
- **A5** I have played a game that uses AI-generated dialogue before. — [Multiple choice] `Yes` / `No` / `Not sure`
- **A6** I am generally positive about AI-generated content in games. — [Linear scale 1–7] *(AI attitude)*
- **A7** I am comfortable reading English text quickly. (optional) — [Linear scale 1–7]
- **A8** In my experience, AI-generated dialogue in games has been disappointing. — [Linear scale 1–7] *(AI attitude — reverse-worded counterpart to A6, deliberately separated from it)*

## Part 2 — After your first detective

> **When:** immediately after the participant submits their verdict for the
> first detective, before they start the second.

- **PID** Participant ID. — [Short answer, required]
- **B0** Which detective did you just interrogate? — [Multiple choice, required] `Detective A` / `Detective B`
- **B1** I was fully focused on the interrogation. — [Linear scale 1–7] *(immersion)*
- **B2** I lost track of time while questioning the suspect. — [Linear scale 1–7] *(immersion — time distortion)*
- **B3** I felt free to ask the suspect whatever I wanted, however I wanted. — [Linear scale 1–7] *(autonomy satisfaction)*
- **B4** I felt restricted in how I could question the suspect. — [Linear scale 1–7] *(autonomy frustration, reversed)*
- **B5** The interrogation felt emotionally tense. — [Linear scale 1–7] *(emotional challenge)*
- **B6** I felt pressure when deciding whether and how to confront the suspect. — [Linear scale 1–7] *(emotional challenge)*
- **B7** The suspect felt like a real character with something at stake. — [Linear scale 1–7] *(believability)*
- **B8** The suspect stayed in character throughout my interrogation. — [Linear scale 1–7] *(persona stability)*
- **B9** The suspect's answers were relevant to what I actually asked. — [Linear scale 1–7] *(coherence)*
- **B10** The suspect reacted believably when I pressed or confronted her. — [Linear scale 1–7] *(reactivity)*
- **B11** I enjoyed playing the role of the interrogator. — [Linear scale 1–7] *(overall PX)*
- **B12** I had enough information to reach a verdict. — [Linear scale 1–7] *(solvability / fairness)*

## Part 3 — After your second detective

> **When:** immediately after the participant submits their verdict for the
> second detective. Covers the second interrogation, the comparison between
> the two, and the closing questions.

### Your second interrogation

Identical to Part 2 — repeat PID and B0–B12 for the detective just interrogated.

- **PID** Participant ID. — [Short answer, required]
- **B0** Which detective did you just interrogate? — [Multiple choice, required] `Detective A` / `Detective B`
- **B1** I was fully focused on the interrogation. — [Linear scale 1–7] *(immersion)*
- **B2** I lost track of time while questioning the suspect. — [Linear scale 1–7] *(immersion — time distortion)*
- **B3** I felt free to ask the suspect whatever I wanted, however I wanted. — [Linear scale 1–7] *(autonomy satisfaction)*
- **B4** I felt restricted in how I could question the suspect. — [Linear scale 1–7] *(autonomy frustration, reversed)*
- **B5** The interrogation felt emotionally tense. — [Linear scale 1–7] *(emotional challenge)*
- **B6** I felt pressure when deciding whether and how to confront the suspect. — [Linear scale 1–7] *(emotional challenge)*
- **B7** The suspect felt like a real character with something at stake. — [Linear scale 1–7] *(believability)*
- **B8** The suspect stayed in character throughout my interrogation. — [Linear scale 1–7] *(persona stability)*
- **B9** The suspect's answers were relevant to what I actually asked. — [Linear scale 1–7] *(coherence)*
- **B10** The suspect reacted believably when I pressed or confronted her. — [Linear scale 1–7] *(reactivity)*
- **B11** I enjoyed playing the role of the interrogator. — [Linear scale 1–7] *(overall PX)*
- **B12** I had enough information to reach a verdict. — [Linear scale 1–7] *(solvability / fairness)*

### Comparing the two detectives

- **C1** Which detective session was the better experience overall? — [Multiple choice] `Detective A` / `Detective B` / `No difference`
- **C2** One of the two suspects was powered by a generative AI; the other used an advanced semantic matching system. Which session do you think had the AI? — [Multiple choice, required] `Detective A` / `Detective B` / `Cannot tell`
- **C3** How confident are you in that guess? — [Linear scale 1–7] *(1 = "Pure guess", 7 = "Completely certain")*
- **C4** What gave it away — or why couldn't you tell? — [Paragraph]
- **C5** Did you notice any difference between the two sessions in how long the suspect took to answer, or in how her text appeared on screen? — [Paragraph]

### Final questions

- **D1** What contributed most to feeling immersed? — [Paragraph]

Name up to 3 things that broke the illusion for you. Leave blank if you have
fewer than three.

- **D2a** 1st thing that broke the illusion. — [Short answer]
- **D2b** 2nd thing that broke the illusion. — [Short answer]
- **D2c** 3rd thing that broke the illusion. — [Short answer]
- **D3** How did you decide on your verdicts? What convinced you? — [Paragraph]
- **D4** Was the session long enough to solve the case? — [Multiple choice, required] `Yes` / `No`
- **D5** Any additional thoughts or comments? — [Paragraph]

## Analysis notes (researchers only)

Instrument sources: B1–B2 PXI immersion subscale; B3–B4 BANGS autonomy;
B5–B6 CORGIS emotional challenge; B11 miniPXI — the instruments used in
Sridharan et al. (CoG 2025) and Ploug et al. (CoG 2025). B7–B10 and C2–C5 are
study-specific (believability + manipulation check of the blind).

- **A6 + A8 are a pair.** A8 is reverse-worded and deliberately separated from
  A6 by an unrelated item, so agreement with both flags straight-lining.
  Reverse-score A8 (8 − x) before averaging, or report the two separately.
- **B8 changed construct.** It previously read "The suspect's answers stayed
  consistent with what she had said earlier" (*consistency* — did she
  contradict herself). It now measures *persona stability* instead. Nothing
  else in the instrument asks about contradiction-spotting, which is arguably
  the sharpest static-vs-dynamic discriminator; B9 (relevance) is the closest
  survivor, and C4 may pick it up in free text. Split B8 back into two items
  if the pilot suggests contradiction is where the conditions diverge.
- **Verdict confidence is not asked here** — the app logs it at verdict time.
- The removed pre-play item "I expect to enjoy this game" (iExpect) means
  there is no baseline expectation covariate for B11; treat B11 as
  between-condition only.
