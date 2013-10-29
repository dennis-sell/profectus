import HMM
import HMM_algorithms

f = open("training.txt", "r")

training_data = []
sentence = []
for line in f:
  if not line.strip():
    training_data.append(sentence)
    sentence = []
  else:
    word_and_tag = line.split()
    if len(word_and_tag) == 2:
      sentence.append((word_and_tag[0], word_and_tag[1]))

print len(training_data)
model = HMM.AppliedHMM(training_data)


example_sent = "The green cat is in a hat ."
example_tags = model.decode(example_sent.split())
print example_tags

