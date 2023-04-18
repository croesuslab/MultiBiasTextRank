from mbtr import MultiBiasTextRank

# ---------------------- INPUTS ----------------------

articles = (
    ("Yoga is a mind-body practice that has been"
        " shown to have numerous health benefits."
        " Regular yoga practice can help reduce stress and anxiety,"
        " improve flexibility and balance, and even lower blood pressure."
        " In addition, yoga has been linked to increased feelings of well-being"
        " and improved mood."),
    # The following article is irrelevant to the query
    ("A new study has found that drinking green tea"
        " can reduce the risk of heart disease."
        " The study, which followed over 100,000 people for several years,"
        " found that those who drank green tea regularly had a lower risk of heart disease"
        " than those who did not."),
    # The following article is in colloquial language
    ("There are a ton of different types of yoga"
        " out there, each with its own cool benefits."
        " Some kinds, like vinyasa or power yoga,"
        " can give you a really good workout and help"
        " make you stronger and healthier. Other types, like restorative or yin yoga,"
        " are more about relaxing and getting rid of stress."),
    # The following article is in formal/technical language,
    ("There are various types of yoga that can provide distinct physiological"
        " and psychological benefits. Dynamic forms, such as vinyasa or power yoga,"
        " can improve muscular strength and cardiovascular health."
        " On the other hand, restorative or yin yoga can promote relaxation"
        " and stress relief, and may enhance parasympathetic activity."),
)

# IC = Information Content, rel. = relatively
queries = (
    # natural query, colloquial, rel. low IC
    "What are the good things about yoga",
    # noun phrase query, technical, rel. very high IC
    "neurophysiological mechanisms underlying the benefits of yoga"
    # natural query, rel. high IC
    "What are the health benefits of yoga?",
    # title query, formal, rel. low IC
    "Yoga's positive impacts",
    # keywords query, technical, rel. high IC
    "physiological psychological benefits yoga",
)

# Example of a desired summary, in terms of specificity/detail.
specificity_examples: tuple[str] = ( # Doesn't have to be on the same topic
    "Daily meditation has been shown to have a multitude of health benefits."
    " Studies have found that regular meditation practice can help reduce"
    " symptoms of anxiety and depression, improve focus and attention, and"
    " even enhance the immune system. Additionally, different types of"
    " meditation, such as mindfulness or loving-kindness meditation,"
    " offer unique benefits, such as improved emotional regulation"
    " and increased feelings of compassion towards oneself and others.",
)

# ---------------------- MODEL AND OUTPUT ----------------------

mbtr = MultiBiasTextRank(
    n_out_sentences      = 3,
    specificity_examples = specificity_examples,
)

summary = mbtr(
    documents = articles,
    queries   = queries,
)
print(summary)
# Outputs:
# [
# 'There are various types of yoga that can provide distinct
#  physiological and psychological benefits.',
# 'Yoga is a mind-body practice that has been shown to have numerous health benefits.',
# 'In addition, yoga has been linked to increased feelings of well-being and improved mood.'
# ]
