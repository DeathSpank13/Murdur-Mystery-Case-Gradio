// static_dialogue.js
// =================
// Browser port of static_dialogue.py: the model-free "Static Control" baseline.
// Responses are chosen by best-match keyword scoring over a fixed set of topic
// branches; each branch walks down an ordered list of pre-written lines on
// repeat visits. static_dialogue.py in the repo root is the canonical version.

// Each branch: "keywords" decide when it is picked, "lines" are delivered in
// order on repeat visits. Branches are scored against the input; ties break by
// definition order (the first-defined branch wins), so more specific topics
// come earlier.
const BRANCHES = {
  identity: {
    keywords: [
      "your name", "what is her name", "her name", "who are you",
      "what should i call you", "call you", "introduce yourself", "introduce",
    ],
    lines: [
      "Eleanor Vance. Though I rather think you knew that before you sat down, Inspector.",
      "Vance. It's my late husband's name, and the name over the gallery door, if that's what you're after.",
      "I've given you my name once already. It hasn't changed.",
    ],
  },
  occupation: {
    keywords: [
      "gallery", "what do you do", "your job", "for a living",
      "occupation", "your work", "your business", "line of work",
    ],
    lines: [
      "I own a gallery in town. Paintings, mostly, and the occasional piece of sculpture when something fine comes along.",
      "It's modest but respectable. Charles and I ran one or two together over the years, before things soured.",
      "My work is hardly the matter at hand, is it.",
    ],
  },
  greeting: {
    keywords: [
      "hello", "good evening", "good morning", "good afternoon",
      "how are you", "how do you do", "how are you feeling", "nice to meet",
    ],
    lines: [
      "Good evening, Inspector. Forgive me if I'm not at my brightest; it has been a dreadful day.",
      "As well as can be expected, with a man dead in my study and a policeman across my table. Do go on.",
      "I'd rather we got to your questions than dwell on pleasantries.",
    ],
  },
  alibi: {
    keywords: [
      "alibi", "where were you", "where was", "at the time",
      "when he died", "when charles died", "your whereabouts",
    ],
    lines: [
      "I was in the drawing room with the other guests for most of the evening. You can ask any of them.",
      "I stepped out once, only to fetch more wine from the cellar. It took a few minutes, no more.",
      "I have already told you where I was. I cannot make the answer more interesting by repeating it.",
    ],
  },
  lastseen: {
    keywords: [
      "last see", "last saw", "when did you last", "last time you saw",
      "see him alive", "saw him alive",
    ],
    lines: [
      "A little after ten, by the study door. He was telling some long story about Venice, as he always did.",
      "I went to see to the wine after that and never spoke to him again. I wish now that I had.",
      "That was the last of it. I've nothing to add to the hour.",
    ],
  },
  party: {
    keywords: [
      "party", "dinner", "the evening", "last night", "gathering",
      "tell me about the night", "what happened",
    ],
    lines: [
      "A small gathering. Old friends, a little wine, far too much talk of business. Charles was in good spirits when he arrived.",
      "It was an ordinary evening until it wasn't. One moment laughter in the drawing room, the next a scream from the study.",
      "I've described the evening as plainly as I can, Inspector.",
    ],
  },
  guests: {
    keywords: [
      "who else", "guests", "anyone else", "who was there",
      "other people", "who attended", "everyone there",
    ],
    lines: [
      "The Harringtons, my business partner Vivian, Charles, and young Daniel who keeps my books. Seven of us, with the staff.",
      "A close little circle. Any of them will tell you the same of the evening, more or less.",
      "That is the whole guest list. There was no one here who oughtn't have been.",
    ],
  },
  argument: {
    keywords: [
      "argue", "argument", "quarrel", "row", "words with", "fight",
      "disagreement that night", "anyone angry",
    ],
    lines: [
      "Daniel and Charles had words over money near the end. Quiet, but I saw Daniel's face.",
      "I'd not make too much of it, though. He's a gentle boy, and Charles could provoke a saint.",
      "I've told you about the only cross words I noticed. There's nothing more to it.",
    ],
  },
  relationship: {
    keywords: [
      "charles", "victim", "know him", "relationship", "friend",
      "how did you know", "the dead man",
    ],
    lines: [
      "Charles was an old friend. We went back twenty years, and did business together on and off for most of them.",
      "We had our disagreements, as old friends do. Nothing that would lead to this.",
      "I would rather not speak ill of the dead, if it's all the same to you.",
    ],
  },
  motive: {
    keywords: [
      "money", "debt", "owe", "owed", "inherit", "insurance", "motive",
      "gain", "payout", "what did you stand to",
    ],
    lines: [
      "He owed me, in fact. A great deal, and he was slow about it. A debt is a reason to keep a man paying, not to harm him.",
      "There's a modest partnership insurance, yes, standard between gallery partners. I'd hardly call it a fortune.",
      "If money is your theory, Inspector, you'll find it points away from me, not toward.",
    ],
  },
  weapon: {
    keywords: [
      "weapon", "knife", "letter opener", "poison", "how did", "killed with",
      "stabbed", "murder weapon",
    ],
    lines: [
      "The letter opener from the study, I'm told. It sat on the desk in plain view of anyone who passed.",
      "I'd not touched it in weeks. It was decorative more than useful, and anyone at the party could have handled it.",
      "I have nothing further to add about objects in my own house.",
    ],
  },
  study_door: {
    keywords: [
      "study door", "out of place", "anything odd", "notice anything",
      "unusual", "the study", "anything strange", "seem wrong",
    ],
    lines: [
      "The study door was shut, which was odd; it's always left open. I noticed it and thought nothing of it.",
      "Perhaps I should have. But one doesn't expect murder behind a closed door in one's own home.",
      "That's the only thing that struck me as amiss. Make of it what you will.",
    ],
  },
  deny: {
    keywords: [
      "you did", "you killed", "guilty", "confess", "murderer",
      "you murdered", "you're lying", "you are lying", "accuse",
    ],
    lines: [
      "That is an outrageous suggestion. I invited the man into my home.",
      "I will not sit here and be accused. I want my solicitor present.",
      "This conversation is over unless you have something resembling evidence.",
    ],
  },
};

// Used when no branch matches, or when a branch's scripted lines are exhausted.
const FALLBACK_LINES = [
  "I'm not sure what you want me to say to that.",
  "Could you be more specific, Inspector?",
  "I've told you everything I can think of.",
  "I'm afraid I don't follow. Ask me plainly and I'll answer plainly.",
];

// Score how well a branch's keywords match the player's text. Each keyword that
// appears contributes its specificity (word count), so "where were you"
// outweighs a bare word like "story".
function scoreBranch(text, keywords) {
  let score = 0;
  for (const keyword of keywords) {
    if (text.includes(keyword)) score += keyword.split(" ").length;
  }
  return score;
}

// Return the next scripted NPC line for the static condition. `visitCounts` is
// a plain object the caller owns and passes back each turn; it is mutated in
// place so the suspect advances through a branch's lines on repeat visits.
export function getResponse(playerInput, visitCounts) {
  const text = playerInput.toLowerCase();

  let bestName = null;
  let bestScore = 0;
  // Object key order preserves insertion order, so the first-defined branch
  // wins any tie.
  for (const [branchName, branch] of Object.entries(BRANCHES)) {
    const score = scoreBranch(text, branch.keywords);
    if (score > bestScore) {
      bestScore = score;
      bestName = branchName;
    }
  }

  if (bestName !== null) {
    const count = visitCounts[bestName] || 0;
    const lines = BRANCHES[bestName].lines;
    const index = Math.min(count, lines.length - 1); // clamp once exhausted
    visitCounts[bestName] = count + 1;
    return lines[index];
  }

  const count = visitCounts._fallback || 0;
  const index = count % FALLBACK_LINES.length;
  visitCounts._fallback = count + 1;
  return FALLBACK_LINES[index];
}
