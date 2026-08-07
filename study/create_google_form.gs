/**
 * Builds the pilot questionnaire as THREE Google Forms automatically —
 * one per moment in the session, so a participant never sees a later
 * question before it applies.
 *
 *   Part 1  before play                (consent, ID, background)
 *   Part 2  after the first detective  (ID, B0, B1-B12)
 *   Part 3  after the second detective (ID, B0, B1-B12, comparison, final)
 *
 * How to use (takes ~2 minutes):
 *   1. Go to https://script.google.com and click "New project".
 *   2. Delete the placeholder code, paste this whole file, and save.
 *   3. Run the function createAllParts (Run button).
 *      The first run asks you to authorize the script with your account.
 *   4. Open View > Logs (or the Execution log): it prints an EDIT url and a
 *      LIVE url for each part. Paste the three LIVE urls into
 *      study/form_links.json, then run `python study/build_study_page.py`
 *      so the public page at docs/study/ links to them.
 *   The forms also appear in your Google Drive.
 *
 * Mirrors study/questionnaire.md. Edit the text there first, then here.
 */

var CONSENT_BLURB =
    'This study compares two versions of an interrogation game. Your gameplay ' +
    'is logged anonymously (transcript, timings, verdicts). You may stop at any ' +
    'time and ask for your data to be deleted. Some details of the study design ' +
    'will be explained only after the session.';

/** Builds all three parts and logs the six URLs in one block. */
function createAllParts() {
  var forms = [
    createPart1_Background(),
    createPart2_FirstDetective(),
    createPart3_SecondDetectiveAndFinal()
  ];
  Logger.log('=== Paste the LIVE urls into study/form_links.json ===');
  for (var i = 0; i < forms.length; i++) {
    Logger.log('Part ' + (i + 1) + ' EDIT: ' + forms[i].getEditUrl());
    Logger.log('Part ' + (i + 1) + ' LIVE: ' + forms[i].getPublishedUrl());
  }
}

// ---- Part 1: consent + ID + background (before play) --------------------
function createPart1_Background() {
  var form = FormApp.create(
      'NPC Interrogation Study — Part 1: Before you play');
  form.setDescription(CONSENT_BLURB);
  form.setProgressBar(true);

  form.addMultipleChoiceItem()
      .setTitle('I have read the description above and consent to take part.')
      .setChoiceValues(['I consent'])
      .setRequired(true);
  addParticipantId(form, 'Participant ID (filled in by the researcher, e.g. P01)');

  form.addPageBreakItem().setTitle('Background');
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
  // A7 sits between the two AI-attitude items on purpose: A8 is the
  // reverse-worded counterpart to A6 and should not be read next to it.
  addScale(form, 'I am comfortable reading English text quickly. (optional)', false);
  addScale(form, 'In my experience, AI-generated dialogue in games has been ' +
                 'disappointing.');
  return form;
}

// ---- Part 2: after the first interrogation ------------------------------
function createPart2_FirstDetective() {
  var form = FormApp.create(
      'NPC Interrogation Study — Part 2: After your first detective');
  form.setDescription(
      'Fill this in right after you submit your verdict for the FIRST ' +
      'detective, before you start the second one.');
  form.setProgressBar(true);

  addParticipantId(form, 'Participant ID');
  addInterrogationBlock(form);
  return form;
}

// ---- Part 3: after the second interrogation + comparison + final --------
function createPart3_SecondDetectiveAndFinal() {
  var form = FormApp.create(
      'NPC Interrogation Study — Part 3: After your second detective');
  form.setDescription(
      'Fill this in right after you submit your verdict for the SECOND ' +
      'detective. It covers that interrogation, how the two compared, and a ' +
      'few closing questions.');
  form.setProgressBar(true);

  addParticipantId(form, 'Participant ID');
  addInterrogationBlock(form);

  // ---- Section C: comparison + blind check ------------------------------
  form.addPageBreakItem().setTitle('Comparing the two detectives');
  form.addMultipleChoiceItem()
      .setTitle('Which detective session was the better experience overall?')
      .setChoiceValues(['Detective A', 'Detective B', 'No difference']);
  form.addMultipleChoiceItem()
      .setTitle('One of the two suspects was powered by a generative AI; the ' +
                'other used an advanced semantic matching system. Which ' +
                'session do you think had the AI?')
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

  // ---- Section D: final questions ---------------------------------------
  form.addPageBreakItem().setTitle('Final questions');
  form.addParagraphTextItem()
      .setTitle('What contributed most to feeling immersed?');
  form.addSectionHeaderItem()
      .setTitle('Name up to 3 things that broke the illusion for you.')
      .setHelpText('Leave blank if you have fewer than three.');
  form.addTextItem().setTitle('1st thing that broke the illusion');
  form.addTextItem().setTitle('2nd thing that broke the illusion');
  form.addTextItem().setTitle('3rd thing that broke the illusion');
  form.addParagraphTextItem()
      .setTitle('How did you decide on your verdicts? What convinced you?');
  form.addMultipleChoiceItem()
      .setTitle('Was the session long enough to solve the case?')
      .setChoiceValues(['Yes', 'No'])
      .setRequired(true);
  form.addParagraphTextItem()
      .setTitle('Any additional thoughts or comments?');
  return form;
}

/** One 12-item per-detective block, prefixed by the "which detective" anchor. */
function addInterrogationBlock(form) {
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
    'The suspect stayed in character throughout my interrogation.',
    'The suspect\'s answers were relevant to what I actually asked.',
    'The suspect reacted believably when I pressed or confronted her.',
    'I enjoyed playing the role of the interrogator.',
    'I had enough information to reach a verdict.'
  ];
  for (var i = 0; i < items.length; i++) {
    addScale(form, items[i]);
  }
}

/** The required ID field that joins all three response sheets together. */
function addParticipantId(form, title) {
  return form.addTextItem().setTitle(title).setRequired(true);
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
