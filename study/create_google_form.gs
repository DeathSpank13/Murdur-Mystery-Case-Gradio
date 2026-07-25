/**
 * Builds the pilot questionnaire as a Google Form automatically.
 *
 * How to use (takes ~2 minutes):
 *   1. Go to https://script.google.com and click "New project".
 *   2. Delete the placeholder code, paste this whole file, and save.
 *   3. Run the function createPilotQuestionnaire (Run button).
 *      The first run asks you to authorize the script with your account.
 *   4. Open View > Logs (or the Execution log): it prints two URLs —
 *      the EDIT url (for you) and the LIVE url (for participants).
 *   The form also appears in your Google Drive.
 *
 * Mirrors study/questionnaire.md. Edit the text there first, then here.
 */

function createPilotQuestionnaire() {
  var form = FormApp.create('NPC Interrogation Study — Pilot Questionnaire');
  form.setDescription(
    'This study compares two versions of an interrogation game. Your gameplay ' +
    'is logged anonymously (transcript, timings, verdicts). You may stop at any ' +
    'time and ask for your data to be deleted. Some details of the study design ' +
    'will be explained only after the session.'
  );
  form.setProgressBar(true);

  // ---- Section 0: consent + ID ------------------------------------------
  form.addMultipleChoiceItem()
      .setTitle('I have read the description above and consent to take part.')
      .setChoiceValues(['I consent'])
      .setRequired(true);
  form.addTextItem()
      .setTitle('Participant ID (filled in by the researcher, e.g. P01)')
      .setRequired(true);

  // ---- Section A: background (before play) ------------------------------
  form.addPageBreakItem().setTitle('Background (before playing)');
  form.addTextItem().setTitle('Age');
  form.addTextItem().setTitle('Gender (optional)');
  form.addTextItem()
      .setTitle('On average, how many hours per week do you play video games?')
      .setRequired(true);
  addScale(form, 'I am familiar with role-playing / detective games.');
  form.addMultipleChoiceItem()
      .setTitle('I have played a game that uses AI-generated dialogue before.')
      .setChoiceValues(['Yes', 'No', 'Not sure']);
  addScale(form, 'I am generally positive about AI-generated content in games.');
  addScale(form, 'I expect to enjoy this game.');
  addScale(form, 'I am comfortable reading English text quickly. (optional)', false);

  // ---- Sections B1 + B2: after each interrogation ------------------------
  addInterrogationBlock(form, 'After your FIRST interrogation');
  addInterrogationBlock(form, 'After your SECOND interrogation');

  // ---- Section C: comparison + blind check -------------------------------
  form.addPageBreakItem().setTitle('Comparing the two detectives');
  form.addMultipleChoiceItem()
      .setTitle('Which detective session did you find more engaging?')
      .setChoiceValues(['Detective A', 'Detective B', 'No difference']);
  form.addMultipleChoiceItem()
      .setTitle('One of the two suspects was powered by a generative AI; the ' +
                'other used pre-written responses. Which session do you think ' +
                'had the AI?')
      .setChoiceValues(['Detective A', 'Detective B', 'Cannot tell'])
      .setRequired(true);
  form.addScaleItem()
      .setTitle('How confident are you in that guess?')
      .setBounds(1, 7)
      .setLabels('Pure guess', 'Completely certain');
  form.addParagraphTextItem()
      .setTitle('What gave it away — or why couldn\'t you tell?');
  form.addParagraphTextItem()
      .setTitle('Did you notice any difference between the two sessions in how ' +
                'long the suspect took to answer, or in how her text appeared ' +
                'on screen?');

  // ---- Section D: final open questions -----------------------------------
  form.addPageBreakItem().setTitle('Final questions');
  form.addParagraphTextItem()
      .setTitle('What contributed most to feeling immersed — and did ' +
                'anything break the illusion?');
  form.addParagraphTextItem()
      .setTitle('How did you decide on your verdicts? What convinced you?');
  form.addParagraphTextItem()
      .setTitle('Was the session long enough to solve the case?');
  form.addParagraphTextItem()
      .setTitle('Was anything about the game or this questionnaire confusing?');

  Logger.log('EDIT url (for you):        ' + form.getEditUrl());
  Logger.log('LIVE url (for participants): ' + form.getPublishedUrl());
}

/** One 12-item per-detective block, prefixed by the "which detective" anchor. */
function addInterrogationBlock(form, title) {
  form.addPageBreakItem().setTitle(title);
  form.addMultipleChoiceItem()
      .setTitle('Which detective did you just interrogate?')
      .setChoiceValues(['Detective A', 'Detective B'])
      .setRequired(true);
  var items = [
    'I was fully focused on the interrogation.',
    'I lost track of time while questioning the suspect.',
    'I felt free to ask the suspect whatever I wanted, however I wanted.',
    'I felt restricted in how I could question the suspect.',
    'The interrogation felt emotionally tense.',
    'I felt pressure when deciding whether and how to confront the suspect.',
    'The suspect felt like a real character with something at stake.',
    'The suspect\'s answers stayed consistent with what she had said earlier.',
    'The suspect\'s answers were relevant to what I actually asked.',
    'The suspect reacted believably when I pressed or confronted her.',
    'I enjoyed this interrogation.',
    'I had enough information to reach a verdict.'
  ];
  for (var i = 0; i < items.length; i++) {
    addScale(form, items[i]);
  }
}

/** 1-7 agree scale; required unless stated otherwise. */
function addScale(form, title, required) {
  var item = form.addScaleItem()
      .setTitle(title)
      .setBounds(1, 7)
      .setLabels('Strongly disagree', 'Strongly agree');
  item.setRequired(required !== false);
  return item;
}
