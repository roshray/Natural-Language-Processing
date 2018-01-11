
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities


df=pd.read_csv('jokes.csv');



x=df['Question'].values.tolist()
y=df['Answer'].values.tolist()

corpus= x+y
  
tok_corp= [nltk.word_tokenize(sent.decode('utf-8')) for sent in corpus]
       
         
model = gensim.models.Word2Vec(tok_corp, min_count=1, size = 300,sg=1,hs=1,negative=0)

model.save('testmodel')
model = gensim.models.Word2Vec.load('testmodel')
#print(model.most_similar('king'))
 #model= gensim.models.Word2Vec( size=64, min_count=5, sg=1, negative=5, workers=2)
#print(model.most_similar('womans'))
questions='questions-words.txt'
def w2v_model_accuracy(model):

    accuracy = model.accuracy(questions)
    
    sum_incorr = len(accuracy[-1]['incorrect'])
    sum_corr = len(accuracy[-1]['correct'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))
    
#print(model.similarity('woman', 'man'))
#print(model.score(["man"]))
print(w2v_model_accuracy(model))

'''
print('King - man + woman:')
print('')
for word, sim in model.most_similar(positive=['woman','king'], negative=['man']):

    print('\"%s\"\t- similarity: %g' % (word, sim))
print('')

print('Similarity between man and queen:')
print(model.similarity('woman', 'man'))
print(model.score(["man"]))
'''