# NPC Interrogation Study — Pilot Questionnaire (N=10)

Ready to transcribe into Google Forms. Each question is annotated with its
Google Forms question type. All Likert items are **Linear scale 1–7**,
labels: 1 = "Strongly disagree", 7 = "Strongly agree".

Tip: instead of entering these by hand, run `create_google_form.gs`
(same folder) in Google Apps Script — it builds the whole form automatically.

Protocol reminder (facilitator-run, like Ploug et al. 2025):
consent + Section A **before play** → play first interrogation → Section B1 →
play second interrogation → Section B2 → Sections C and D → debrief (reveal
which detective was which).

Facilitator rules:
1. **Sequential interrogations.** The app lets participants switch detectives
   per question — instruct them to fully interrogate one detective and submit
   its verdict before starting the other, so the B1/B2 blocks stay clean.
2. **Counterbalance order to 5/5.** The label↔condition mapping is randomized
   once per session (`new_label_mapping()` in ui.py). Peek at researcher view
   (off the participant's screen) and tell them which detective to start with,
   so 5 participants meet the static suspect first and 5 the dynamic one.
3. **Linkage.** Note each participant's session log filename next to their
   Participant ID. Every logged turn and verdict carries both `condition` and
   `label_shown`, so B0 + the log resolves which block was which condition.

---

## Section 0 — Consent and ID

> This study compares two versions of an interrogation game. Your gameplay is
> logged anonymously (transcript, timings, verdicts). You may stop at any time
> and ask for your data to be deleted. Some details of the study design will
> be explained only after the session.

- **0.1** I have read the above and consent to take part. — [Multiple choice, required] `I consent`
- **0.2** Participant ID (filled in by the researcher, e.g. P01). — [Short answer, required]

## Section A — Background (before play)

- **A1** Age. — [Short answer]
- **A2** Gender (optional). — [Short answer]
- **A3** On average, how many hours per week do you play video games? — [Short answer, number]
- **A4** I am familiar with role-playing / detective games. — [Linear scale 1–7]
- **A5** I have played a game that uses AI-generated dialogue before. — [Multiple choice] `Yes` / `No` / `Not sure`
- **A6** I am generally positive about AI-generated content in games. — [Linear scale 1–7]
- **A7** I expect to enjoy this game. — [Linear scale 1–7]
- **A8** I am comfortable reading English text quickly. (optional) — [Linear scale 1–7]

## Section B1 — After your FIRST interrogation

- **B0** Which detective did you just interrogate? — [Multiple choice, required] `Detective A` / `Detective B`

All below [Linear scale 1–7]:

- **B1** I was fully focused on the interrogation. *(immersion)*
- **B2** I lost track of time while questioning the suspect. *(immersion — time distortion; suits a text-only UI better than sensory-immersion wording)*
- **B3** I felt free to ask the suspect whatever I wanted, however I wanted. *(autonomy satisfaction)*
- **B4** I felt restricted in how I could question the suspect. *(autonomy frustration, reversed)*
- **B5** The interrogation felt emotionally tense. *(emotional challenge)*
- **B6** I felt pressure when deciding whether and how to confront the suspect. *(emotional challenge)*
- **B7** The suspect felt like a real character with something at stake. *(believability)*
- **B8** The suspect's answers stayed consistent with what she had said earlier. *(consistency)*
- **B9** The suspect's answers were relevant to what I actually asked. *(coherence)*
- **B10** The suspect reacted believably when I pressed or confronted her. *(reactivity)*
- **B11** I enjoyed this interrogation. *(overall PX)*
- **B12** I had enough information to reach a verdict. *(solvability / fairness)*

## Section B2 — After your SECOND interrogation

Identical to Section B1 (repeat B0–B12).

## Section C — Comparing the two detectives (after both)

- **C1** Which detective session did you find more engaging? — [Multiple choice] `Detective A` / `Detective B` / `No difference`
- **C2** One of the two suspects was powered by a generative AI; the other used pre-written responses. Which session do you think had the AI? — [Multiple choice, required] `Detective A` / `Detective B` / `Cannot tell`
- **C3** How confident are you in that guess? — [Linear scale 1–7] (1 = "Pure guess", 7 = "Completely certain")
- **C4** What gave it away — or why couldn't you tell? — [Paragraph]
- **C5** Did you notice any difference between the two sessions in how long the suspect took to answer, or in how her text appeared on screen? — [Paragraph]

## Section D — Final questions

- **D1** What contributed most to feeling immersed — and did anything break the illusion? — [Paragraph]
- **D2** How did you decide on your verdicts? What convinced you? — [Paragraph]
- **D3** Was the session long enough to solve the case? — [Paragraph]
- **D4** Was anything about the game or this questionnaire confusing? — [Paragraph]

---

Instrument sources: B1–B2 PXI immersion subscale; B3–B4 BANGS autonomy;
B5–B6 CORGIS emotional challenge; B11 miniPXI; A7 iExpect — the instruments
used in Sridharan et al. (CoG 2025) and Ploug et al. (CoG 2025). B7–B10 and
C2–C5 are study-specific (believability + manipulation check of the blind).
Verdict confidence is NOT asked here — the app logs it at verdict time.
