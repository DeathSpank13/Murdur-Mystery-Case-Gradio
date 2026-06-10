// branching_dialogue.js
// ====================
// Browser port of branching_dialogue.py: the clickable choice-tree mode. The
// player picks from a menu instead of typing. Three consequence rules are
// enforced by the engine:
//   once             a question consumed after first use
//   exclusiveGroup   a fork; choosing one option locks its whole group
//   goto / nesting   descend into a child node; a synthetic Back climbs out
//
// branching_dialogue.py in the repo root is the canonical version.

const BACK_ID = "__back__";
const BACK_TEXT = "← Back";

// Each node has an "intro" (what Eleanor says on arrival) and ordered "options".
// An option: { id, text, response?, once?, exclusiveGroup?, goto? }
// (Python used snake_case "exclusive_group"; here it is exclusiveGroup.)
export const DIALOGUE_TREE = {
  main: {
    intro:
      "Of course, Inspector. Ask me whatever you need. It has been a dreadful " +
      "day, but I'd like to help you make sense of it. Where shall we start?",
    options: [
      {
        id: "party",
        text: "Tell me about last night's party.",
        response:
          "A small gathering. Old friends, a little wine, far too much talk of " +
          "business. Charles was in good spirits when he arrived.",
        goto: "party",
      },
      {
        id: "alibi",
        text: "Where were you when Charles died?",
        response:
          "In the drawing room with my guests, for most of the evening. I " +
          "stepped out once for wine, no more than a few minutes. Ask any of them.",
        once: true,
        goto: "alibi",
      },
      {
        id: "relationship",
        text: "What was your relationship with Charles?",
        response:
          "We went back twenty years. Friends, and partners in a gallery or " +
          "two. Friendship like that has its weather, Inspector.",
        goto: "relationship",
      },
      {
        id: "weapon",
        text: "Let's talk about the murder weapon.",
        response:
          "The letter opener from the study, I'm told. A dreadful thing to " +
          "imagine. It sat on the desk in plain view of anyone who passed.",
        goto: "weapon",
      },
      {
        id: "press",
        text: "I think you're hiding something. (press her hard)",
        response:
          "Hiding something? I open my home, lose a friend, and am rewarded " +
          "with accusations. Tread carefully, Inspector.",
        exclusiveGroup: "approach",
        goto: "press",
      },
      {
        id: "reassure",
        text: "You're not a suspect. Help me understand. (reassure her)",
        response:
          "Thank you. It has been a horrid day, and to be treated gently is a " +
          "kindness. Ask me anything; I want him found out as much as you do.",
        exclusiveGroup: "approach",
        goto: "reassure",
      },
    ],
  },

  party: {
    intro: "What is it about the evening you'd like to know?",
    options: [
      {
        id: "guests",
        text: "Who else was at the party?",
        response:
          "The Harringtons, my business partner Vivian, Charles, and young " +
          "Daniel who keeps my books. Seven of us, with the staff.",
      },
      {
        id: "argument",
        text: "Did anyone argue with Charles that night?",
        response:
          "Daniel and he had words over money near the end. Quiet, but I saw " +
          "Daniel's face. I'd not make too much of it, though. He is a gentle boy.",
        once: true,
      },
      {
        id: "lastseen",
        text: "When did you last see Charles alive?",
        response:
          "A little after ten, by the study door, telling some long story " +
          "about Venice. I went to see to the wine and never spoke to him again.",
        once: true,
      },
    ],
  },

  alibi: {
    intro: "My whereabouts. Press me on it if you must.",
    options: [
      {
        id: "wine_cellar",
        text: "Tell me about the trip to the cellar.",
        response:
          "Down the back stairs, two bottles of the Margaux, back up. Five " +
          "minutes, perhaps seven. The cellar is cold; one doesn't linger.",
        once: true,
      },
      {
        id: "witness",
        text: "Can anyone confirm you were in the drawing room?",
        response:
          "Vivian, certainly. She and I were thick as thieves on the settee " +
          "most of the night. Though she did step out for some air around then, " +
          "now I think of it.",
      },
    ],
  },

  relationship: {
    intro: "Charles and I. Where shall I start?",
    options: [
      {
        id: "business",
        text: "You did business together?",
        response:
          "Three galleries over the years. The last one did poorly, and money " +
          "has a way of souring even old affection.",
        goto: "business",
      },
      {
        id: "disagreement",
        text: "You mentioned disagreements. About what?",
        response:
          "The usual things. He thought me reckless with the accounts; I " +
          "thought him a coward with them. We were both a little right.",
        once: true,
      },
    ],
  },

  business: {
    intro: "The business, then. It's no secret it ended badly.",
    options: [
      {
        id: "debt",
        text: "Did Charles owe you money, or you him?",
        response:
          "He owed me. A great deal, and he was slow about it. I'd have been " +
          "paid eventually. I am not a fool about these things.",
        once: true,
      },
      {
        id: "insurance",
        text: "Was there any insurance or payout tied to him?",
        response:
          "On the gallery partnership, yes, a modest one. Standard between " +
          "partners. I'd hardly call it a fortune.",
        once: true,
      },
    ],
  },

  weapon: {
    intro: "That wretched letter opener. What of it?",
    options: [
      {
        id: "who_handled",
        text: "Who could have handled it?",
        response:
          "Anyone. It lived on the study desk. I'd not touched it in weeks. It " +
          "was decorative more than useful.",
      },
      {
        id: "prints",
        text: "Whose fingerprints would we expect to find on it?",
        response:
          "Mine, I suppose, from dusting the desk. And half the county's, for " +
          "all I know. It's hardly under lock and key.",
        once: true,
      },
    ],
  },

  press: {
    intro: "Go on, then. Bully me with your theories and see where it gets you.",
    options: [
      {
        id: "accuse_direct",
        text: "You killed Charles, didn't you?",
        response:
          "How dare you. I want my solicitor, and I want this conversation " +
          "noted, every word of it. I'll not say another thing without counsel.",
        once: true,
      },
      {
        id: "motive_money",
        text: "He owed you money. That's a motive.",
        response:
          "A debt is a reason to keep a man alive and paying, Inspector, not to " +
          "put a blade in him. Do think it through.",
        once: true,
      },
    ],
  },

  reassure: {
    intro: "You're kind to say so. What can I tell you that would help?",
    options: [
      {
        id: "who_suspect",
        text: "Who do you think could have done this?",
        response:
          "If I had to point a finger, and I hate to do it, I would look at " +
          "Daniel and that quarrel over money. But I may be wronging the boy.",
        once: true,
      },
      {
        id: "anything_odd",
        text: "Did anything seem out of place that night?",
        response:
          "The study door was shut when it's always left open. I noticed it and " +
          "thought nothing of it. Perhaps I should have.",
        once: true,
      },
    ],
  },
};

// Walks DIALOGUE_TREE for one run and tracks the consequence state. The
// transcript is a list of { speaker, text } where speaker is "npc" or "player".
export class DialogueEngine {
  constructor(tree = DIALOGUE_TREE, start = "main") {
    this.tree = tree;
    this.current = start;
    this.consumed = new Set();      // ids of one-time options already used
    this.lockedGroups = new Set();  // exclusive groups already committed to
    this.chosen = new Set();        // every option ever picked (for UI dimming)
    this.stack = [];                // node ids to climb back through
    this.transcript = [];

    const intro = this.tree[start].intro;
    if (intro) this.transcript.push({ speaker: "npc", text: intro });
  }

  // Options visible at the current node right now: drops used one-time options
  // and options whose exclusive group is locked, and appends Back when there is
  // somewhere to climb to.
  availableOptions() {
    const options = [];
    for (const opt of this.tree[this.current].options) {
      if (opt.once && this.consumed.has(opt.id)) continue;
      if (opt.exclusiveGroup && this.lockedGroups.has(opt.exclusiveGroup)) continue;
      options.push(opt);
    }
    if (this.stack.length > 0) options.push({ id: BACK_ID, text: BACK_TEXT });
    return options;
  }

  // Apply the option with the given id and return it. Throws if the id is not
  // currently available, so stale-id UI bugs surface loudly.
  choose(optionId) {
    if (optionId === BACK_ID) {
      if (this.stack.length > 0) {
        this.current = this.stack.pop();
        const intro = this.tree[this.current].intro;
        if (intro) this.transcript.push({ speaker: "npc", text: intro });
      }
      return { id: BACK_ID, text: BACK_TEXT };
    }

    const option = this._findAvailable(optionId);

    this.transcript.push({ speaker: "player", text: option.text });
    if (option.response) {
      this.transcript.push({ speaker: "npc", text: option.response });
    }

    this.chosen.add(optionId);
    if (option.once) this.consumed.add(optionId);
    if (option.exclusiveGroup) this.lockedGroups.add(option.exclusiveGroup);

    if (option.goto) {
      this.stack.push(this.current);
      this.current = option.goto;
      const intro = this.tree[option.goto].intro;
      if (intro) this.transcript.push({ speaker: "npc", text: intro });
    }

    return option;
  }

  _findAvailable(optionId) {
    for (const opt of this.availableOptions()) {
      if (opt.id === optionId) return opt;
    }
    throw new Error(
      `Option '${optionId}' is not available at node '${this.current}'.`
    );
  }
}

export { BACK_ID };
