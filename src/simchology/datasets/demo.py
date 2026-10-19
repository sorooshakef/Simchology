"""A small synthetic dataset for trying out Simchology (German or English).

Participants "wrote" short texts about one of three topics; each text is paired with
every reference information ("info") of its topic, in long format, as in typical
information-recall studies. Labels say whether the info is contained in the text
(``label``: 0/1) and how fully (``label_gradient``: 0, 0.5 for a vague mention without
the actual claim, 1). About 10% of the texts lack their final punctuation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# topic -> list of (info, [paraphrases], vague mention)
_DE = {
    "Giraffen": [
        ("Giraffen sind die höchsten Landtiere der Welt.",
         ["Keine anderen Landtiere der Welt sind so hoch wie Giraffen.",
          "Die Giraffe ist das höchste Tier, das an Land lebt."],
         "Über die Größe von Giraffen weiß ich nichts mehr."),
        ("Eine Giraffe hat genauso viele Halswirbel wie ein Mensch, nämlich sieben.",
         ["Giraffen haben sieben Halswirbel, also so viele wie Menschen.",
          "Trotz des langen Halses besitzt die Giraffe nur sieben Halswirbel wie der Mensch."],
         "Der Hals der Giraffe hat irgendwie Wirbel."),
        ("Giraffen schlafen nur etwa zwei Stunden am Tag.",
         ["Am Tag schlafen Giraffen bloß ungefähr zwei Stunden.",
          "Eine Giraffe braucht nur rund zwei Stunden Schlaf pro Tag."],
         "Wie lange Giraffen schlafen, habe ich vergessen."),
        ("Die Zunge einer Giraffe ist bis zu 50 Zentimeter lang.",
         ["Giraffen haben eine Zunge, die bis zu 50 Zentimeter lang werden kann.",
          "Mit ihrer bis zu 50 Zentimeter langen Zunge greifen Giraffen nach Blättern."],
         "Die Zunge der Giraffe ist auffällig."),
        ("Giraffen fressen hauptsächlich Blätter von Akazien.",
         ["Die Hauptnahrung der Giraffen sind Blätter von Akazien.",
          "Am liebsten fressen Giraffen die Blätter der Akazie."],
         "Giraffen fressen Pflanzen."),
    ],
    "Moai": [
        ("Die Moai sind riesige Steinstatuen auf der Osterinsel.",
         ["Auf der Osterinsel stehen riesige Statuen aus Stein, die Moai genannt werden.",
          "Moai heißen die großen steinernen Statuen der Osterinsel."],
         "Die Osterinsel ist sehr abgelegen."),
        ("Die meisten Moai wurden aus Vulkangestein gehauen.",
         ["Man hat die meisten Moai aus Vulkangestein gemeißelt.",
          "Das Material der meisten Moai ist Vulkangestein."],
         "Auf der Insel gibt es Vulkane."),
        ("Es gibt fast 900 Moai auf der Insel.",
         ["Auf der Insel befinden sich beinahe 900 Moai.",
          "Insgesamt zählt man auf der Insel fast 900 dieser Statuen."],
         "Es gibt sehr viele Statuen."),
        ("Die Statuen blicken meist ins Landesinnere.",
         ["Meistens blicken die Statuen nicht aufs Meer, sondern ins Landesinnere.",
          "Die Statuen sind überwiegend zum Landesinneren ausgerichtet."],
         "Wohin die Statuen blicken, weiß ich nicht."),
        ("Die Moai stellen vergöttlichte Vorfahren dar.",
         ["Die Statuen zeigen die vergöttlichten Vorfahren der Bewohner.",
          "Mit den Moai wurden verstorbene Vorfahren als Götter dargestellt."],
         "Die Bedeutung der Moai ist umstritten."),
    ],
    "Römische Wasserversorgung": [
        ("Die Römer bauten Aquädukte, um Wasser in die Städte zu leiten.",
         ["Mit Aquädukten leiteten die Römer Wasser in ihre Städte.",
          "Um Wasser in die Städte zu bringen, bauten die Römer Aquädukte."],
         "Die Römer bauten viel."),
        ("Das Wasser floss allein durch das Gefälle.",
         ["Allein durch das Gefälle floss das Wasser vorwärts.",
          "Pumpen waren nicht nötig, weil das Wasser durch das Gefälle floss."],
         "Das Wasser floss irgendwie."),
        ("Rom wurde von elf großen Aquädukten versorgt.",
         ["Elf große Aquädukte versorgten die Stadt Rom.",
          "Die Stadt Rom wurde über elf große Aquädukte mit Wasser versorgt."],
         "Rom war eine große Stadt."),
        ("Viele Häuser hatten Wasserleitungen aus Blei.",
         ["In vielen Häusern gab es Wasserleitungen aus Blei.",
          "Die Wasserleitungen vieler Häuser bestanden aus Blei."],
         "Die Häuser waren unterschiedlich."),
        ("Öffentliche Brunnen versorgten die einfache Bevölkerung mit Wasser.",
         ["Die einfache Bevölkerung holte ihr Wasser an öffentlichen Brunnen.",
          "Für die einfache Bevölkerung gab es öffentliche Brunnen, die sie mit Wasser versorgten."],
         "Wasser war für alle wichtig."),
    ],
}

_EN = {
    "Giraffes": [
        ("Giraffes are the tallest land animals in the world.",
         ["No other land animal in the world is as tall as the giraffe.",
          "The giraffe is the tallest animal living on land."],
         "I no longer remember anything about how big giraffes are."),
        ("A giraffe has as many neck vertebrae as a human, namely seven.",
         ["Giraffes have seven neck vertebrae, just as many as humans.",
          "Despite its long neck, the giraffe has only seven neck vertebrae like a human."],
         "The giraffe's neck somehow has vertebrae."),
        ("Giraffes sleep only about two hours a day.",
         ["Giraffes sleep for only around two hours per day.",
          "A giraffe needs just about two hours of sleep a day."],
         "I forgot how long giraffes sleep."),
        ("A giraffe's tongue can be up to 50 centimetres long.",
         ["Giraffes have a tongue that can grow up to 50 centimetres long.",
          "With their tongue of up to 50 centimetres, giraffes grab leaves."],
         "The giraffe's tongue is remarkable."),
        ("Giraffes mainly eat the leaves of acacia trees.",
         ["The main food of giraffes is the leaves of acacia trees.",
          "Giraffes like to eat acacia leaves most of all."],
         "Giraffes eat plants."),
    ],
    "Moai": [
        ("The moai are huge stone statues on Easter Island.",
         ["On Easter Island there are huge statues made of stone, called moai.",
          "The moai are the large stone statues of Easter Island."],
         "Easter Island is very remote."),
        ("Most moai were carved from volcanic rock.",
         ["Most of the moai were carved out of volcanic rock.",
          "The material of most moai is volcanic rock."],
         "There are volcanoes on the island."),
        ("There are almost 900 moai on the island.",
         ["The island has nearly 900 moai.",
          "In total, almost 900 of these statues stand on the island."],
         "There are very many statues."),
        ("The statues mostly face inland.",
         ["The statues mostly do not face the sea but look inland.",
          "Most of the statues are turned to face inland."],
         "I do not know which way the statues face."),
        ("The moai represent deified ancestors.",
         ["The statues show the deified ancestors of the islanders.",
          "The moai depicted dead ancestors as gods."],
         "The meaning of the moai is disputed."),
    ],
    "Roman water supply": [
        ("The Romans built aqueducts to carry water into the cities.",
         ["The Romans used aqueducts to carry water into their cities.",
          "To bring water into the cities, the Romans built aqueducts."],
         "The Romans built a lot."),
        ("The water flowed by gravity alone.",
         ["Gravity alone made the water flow.",
          "No pumps were needed because the water flowed by gravity."],
         "The water flowed somehow."),
        ("Rome was supplied by eleven large aqueducts.",
         ["Eleven large aqueducts supplied the city of Rome.",
          "The city of Rome got its water from eleven large aqueducts."],
         "Rome was a big city."),
        ("Many houses had water pipes made of lead.",
         ["Many houses had pipes made of lead for their water.",
          "The water pipes of many houses were made of lead."],
         "The houses were all different."),
        ("Public fountains supplied ordinary people with water.",
         ["Ordinary people got their water from public fountains.",
          "Public fountains supplied water to the ordinary population."],
         "Water was important for everyone."),
    ],
}

_FILLERS = {
    "de": ["Ich fand den Text spannend.", "Mehr weiß ich leider nicht mehr.",
           "Der Text war ziemlich lang.", "Das Thema war mir vorher unbekannt.",
           "Ich habe den Rest vergessen."],
    "en": ["I found the text interesting.", "Unfortunately I do not remember more.",
           "The text was quite long.", "I did not know much about the topic before.",
           "I forgot the rest."],
}


def load_demo(language: str = "de", n_per_topic: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic long-format coded data: one row per (participant text, reference info).

    Columns: ``ID``, ``participant``, ``topic``, ``text``, ``info``, ``label`` (0/1:
    info contained) and ``label_gradient`` (0 / 0.5 vague mention / 1).
    """
    if language not in ("de", "en"):
        raise ValueError("language must be 'de' or 'en'")
    topics = _DE if language == "de" else _EN
    fillers = _FILLERS[language]
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for topic, infos in topics.items():
        for _ in range(n_per_topic):
            pid += 1
            k = int(rng.integers(1, 4))
            included = set(rng.choice(len(infos), size=k, replace=False).tolist())
            others = [i for i in range(len(infos)) if i not in included]
            vague = set(rng.choice(others, size=int(rng.integers(0, 2)), replace=False).tolist()) if others else set()
            sentences = []
            for i in included:
                info, paraphrases, _ = infos[i]
                sentences.append(info if rng.random() < 0.2 else paraphrases[int(rng.integers(0, len(paraphrases)))])
            sentences += [infos[i][2] for i in vague]
            sentences += list(rng.choice(fillers, size=int(rng.integers(1, 3)), replace=False))
            rng.shuffle(sentences)
            text = " ".join(sentences)
            if rng.random() < 0.1:
                text = text.rstrip(".!?")
            for i, (info, _, _) in enumerate(infos):
                rows.append({
                    "participant": f"P{pid:03d}",
                    "topic": topic,
                    "text": text,
                    "info": info,
                    "label": int(i in included),
                    "label_gradient": 1.0 if i in included else (0.5 if i in vague else 0.0),
                })
    df = pd.DataFrame(rows)
    df.insert(0, "ID", np.arange(1, len(df) + 1))
    return df
