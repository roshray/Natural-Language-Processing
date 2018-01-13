
// Start H2O services
import org.apache.spark.h2o._
val h2oContext = H2OContext.getOrCreate(spark)

import _root_.hex.word2vec.{Word2Vec, Word2VecModel}
import _root_.hex.word2vec.Word2VecModel.Word2VecParameters
import water.fvec.Vec

// strings which are not usefull for text mining
val STOP_WORDS = Set("the", "a", "and", "of", "was")

// use the built in tokenizer
val sparkframe = sc.parallelize(Array("I went to the sweet shop where there was lots of treats and chocolates like lollipops, bonbons, and starbursts"))

def tokenize(line: String) = {
  //get rid of nonWords such as punctuation as opposed to splitting by just " "
  line.split("""\W+""")
    .map(_.toLowerCase)

    //remove stopwords defined above (you can add to this list if you want)
    .filterNot(word => STOP_WORDS.contains(word)) :+ null
}

// if using your own tokenizer function, then map function to DataFrame
val allLabelledWords = sparkframe.flatMap(d => tokenize(d))

// convert to H2OFrame
val h2oFrame = h2oContext.asH2OFrame(allLabelledWords)

// specify the parameters for the word2vec model
val w2vParams = new Word2VecParameters
w2vParams._train = h2oFrame._key
w2vParams._epochs = 5
w2vParams._min_word_freq = 0
w2vParams._init_learning_rate = 0.05f
w2vParams._window_size = 5
w2vParams._vec_size = 5

val w2v = new Word2Vec(w2vParams).trainModel().get()
// Find synonyms using a word2vec model.
// pass in word with a single word to find synonyms for 
// pass in the top 'count' synonyms to return
w2v.findSynonyms("lollipops", 3)

// transform words using w2v model and aggregate method average
// he transform(..) function takes an H2O Vec as the first parameter, 
// the vector needs to be extracted from the H2O frame h2oFrame.
val newSparkFrame = w2v.transform(h2oFrame.vec(0), Word2VecModel.AggregateMethod.NONE).toTwoDimTable()




