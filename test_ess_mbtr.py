from mbtr import MultiBiasTextRank

# ---------------------- INPUTS ----------------------

queries = (
    "Roomba",
    "Amazon store",
    "Canada",
)
articles = (
    # Positive feedback 1
    " The Canadian Roomba from Amazon effortlessly navigates around my house,"
    " picking up dirt and dust in every nook and cranny."
    " The battery life is impressive and it's so easy to clean and maintain."
    " I love being able to control it from my phone and schedule cleaning sessions while I'm out of"
    " the house. I highly recommend this Roomba to anyone looking for a reliable and effective"
    " cleaning solution!",
    # Positive feedback 2
    "I can't believe how much time and effort the Canadian Roomba from Amazon has saved me!"
    " As a busy professional, I don't have time to spend hours vacuuming"
    " every week, but this littlerobot has made my life so much easier."
    " It's smart enough to avoid obstacles and navigate"
    " around furniture, and it does an incredible job of picking up pet hair and other debris."
    " I also appreciate how quiet it is compared to other vacuums I've used in the past."
    " Overall, I'm extremely satisfied with my purchase"
    " and would definitely recommend it to others!",
    # Feedback on a different product, different online store
    "I've always been a fan of multi-functional gadgets, so I was really excited to try"
    " out the new 3-in-1 blender that I had ordered from eBay."
    " Unfortunately, my experience with it has been pretty underwhelming. It is quite"
    " loud and doesn't seem to mix ingredients as smoothly as I had hoped.",
    # Negative feedback 1
    "I had high hopes for the Canadian Roomba I ordered from Amazon, but unfortunately it just"
    " didn't live up to my expectations. The suction power is weak and it"
    " struggles to pick up larger debris like crumbs or pet food."
    " It also has difficulty navigating around my house and"
    " often gets stuck under furniture or in tight spaces. I find myself having to constantly"
    " rescue it and move it to a different area. Overall, I'm disappointed in this purchase and"
    " would not recommend it.",
    # Negative feedback 2
    "I had high expectations for the Canadian Roomba I purchased from Amazon,"
    " but it didn't work well on my carpets. It struggles to pick up pet hair and often leaves"
    " clumps of it behind. It also doesn't do a good job of cleaning around the edges of the room,"
    " so I still have to go back and manually vacuum those areas. To top it off, it's very loud"
    " and disruptive when it's running. Overall, I'm disappointed in the performance of this"
    " product and would not recommend it to others.",
)

# ---------------------- MODEL AND OUTPUT ----------------------

mbtr = MultiBiasTextRank(n_out_sentences = 3)

positive_summary = mbtr(
    documents = articles,
    queries   = queries,
    sentiment = True,
)
print(positive_summary)
# Outputs:
# [
# 'The Canadian Roomba from Amazon effortlessly navigates around my house,
#  picking up dirt and dust in every nook and cranny.',
# 'I highly recommend this Roomba to anyone looking for a
#  reliable and effective cleaning solution!',
# "I can't believe how much time and effort the Canadian Roomba from Amazon has saved me!"
# ]

negative_summary = mbtr(
    documents = articles,
    queries   = queries,
    sentiment = False,
)
print(negative_summary)
# Outputs:
# [
# "I had high hopes for the Canadian Roomba I ordered from Amazon,
#  but unfortunately it just didn't live up to my expectations.",
# "I had high expectations for the Canadian Roomba I purchased from Amazon,
#  but it didn't work well on my carpets.",
# "I can't believe how much time and effort the Canadian Roomba from Amazon has saved me!"
# ]
