from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Bangladesh,[a] officially the People's Republic of Bangladesh,[b] is a country in South Asia. It is the eighth-most populous country in the world with a population of almost 177.8 million people within an area of 148,461 square kilometres (57,321 sq mi).[9] Bangladesh shares land borders with India to the north, west, and east, and Myanmar to the southeast. It has a coastline along the Bay of Bengal to its south and is separated from Bhutan and Nepal by the Siliguri Corridor, and from China by the Indian states of West Bengal and Sikkim to its north. Dhaka, the capital and largest city, is the nation's political, financial, and cultural centre. Chittagong is the second-largest city and the busiest port of the country.

The territory of modern Bangladesh was ruled by successive Hindu and Buddhist dynasties in ancient history. Following the Muslim conquest in 1204, the region came under Sultanate and Mughal rule. As the largest subdivision of the Mughal Empire, Bengal became a major commercial centre, known for its textile industry and agricultural output. The Battle of Plassey in 1757 began British colonial rule, which lasted nearly two centuries; colonial policies redirected wealth from Bengal to Britain, while the region itself suffered repeated deadly famines. After the Partition of India in 1947, East Bengal became the eastern wing of the newly formed Dominion of Pakistan and was later renamed East Pakistan.[15]
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

data = splitter.split_text(text)

for chunk in data:
    print(chunk)
    print("\n==================\n")
