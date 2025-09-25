import random

colors = [
    "red", "white", "black", "purple", "pink", "yellow", "green", "blue",
    "orange", "brown"
]

objects = [
    "car", "dog", "apple", "banana", "cup", "sandwich", "giraffe", "backpack",
    "sheep", "bird", "bear", "cell phone", "book", "vase", "hat", "clock",
    "chair", "table", "cat", "shoe", "laptop", "bottle", "pen", "notebook",
    "pillow", "blanket", "mug", "camera", "tree", "bench"
]

def create_phrase():
    if random.random() < 0.5:
        color = random.choice(colors)
        obj = random.choice(objects)
        article = "An" if color[0] in "aeiou" else "A"
        return f"{article} {color} colored {obj}."
    else:
        c1, c2 = random.sample(colors, 2)
        o1, o2 = random.sample(objects, 2)
        article1 = "An" if c1[0] in "aeiou" else "A"
        article2 = "an" if c2[0] in "aeiou" else "a"
        return f"{article1} {c1} {o1} and {article2} {c2} {o2}."

# Use a set to avoid duplicates
unique_prompts = set()
while len(unique_prompts) < 1200:
    phrase = create_phrase()
    unique_prompts.add(phrase)

# Convert to sorted list for consistency (optional)
prompts = list(unique_prompts)

# Save to file
with open("testing_prompts.txt", "w") as f:
    for prompt in prompts:
        f.write(prompt + "\n")

print("1200 unique prompts saved to testing_prompts.txt")
