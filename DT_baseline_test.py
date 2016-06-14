import re, nltk, random, sys
import word_category_counter
from nltk import bigrams

selected_features = None

def get_score(review):
    return int(re.search(r'Overall = ([1-5])', review).group(1))

def get_text(review):
    return re.search(r'Text = "(.*)"', review).group(1)
		
# Write to File, this function is just for reference, because the encoding matters.
def write_file(file_name, data):
    file = open(file_name, 'w', encoding="utf-8")    # or you can say encoding="latin1"
    file.write(data)
    file.close()


def process_reviews(file_name):
    file = open(file_name, "rb")
    raw_data = file.read().decode("latin1")
    file.close()
    
    stopwords = nltk.corpus.stopwords.words("english")
    np = r"(\w)"

    texts = []
    for review in re.split(r'\.\n', raw_data):
        overall_score = get_score(review)
        review_text = get_text(review)
        if overall_score > 3:
            score = "positive"
        else: 
        	score = "negative"
        	
        words = nltk.word_tokenize(review_text)
        flat_words = [word.lower() for word in words]
        #print(flat_words)
        content = []
        for t in  flat_words:
        	if t not in stopwords:
        		content.append(t)
        #print(content)
        content_w = []
        for t in content:
        	result = re.search(np, t)
        	if result != None:
        		content_w.append(t)
        	        
        item = (review_text, content_w, score)
        texts.append(item)
        
    return texts
    		
# Write to File, this function is just for reference, because the encoding matters.
def write_file(file_name, data):
    file = open(file_name, 'w', encoding="utf-8")    # or you can say encoding="latin1"
    file.write(data)
    file.close()

# Adds unigram based lexical features
def add_lexical_features(fdist, bidist, feature_vector, text):
	text_t = nltk.word_tokenize(text)
	text_nl = nltk.Text(text_t)
	text_len = len(text_nl)
	
	#unigram features
	for word, freq in fdist.items(): # fdist = nltk.FreqDist(review_words)
		fname = "UNI_" + word
        
        # If we haven't selected any features yet then add the feature to
        # our feature vector
        # Otherwise make sure the feature is one of the ones we want
        # Note we use a Set for the selected features for O(1) lookup
		if selected_features == None or fname in selected_features:
            #feature_vector[fname] = 1            
			#num_of_w = text_nl.count(word)
			feature_vector[fname] = fdist.freq(word)
	
	#let's add a new feature "len of text"
	fname = "text_len"
	feature_vector[fname] = text_len
	
	#bigram features
	for word, freq in bidist.items(): # fdist = nltk.FreqDist(review_words)
		fname = "BIGRAM_" + word[0] + "_" + word[1]
		if selected_features == None or fname in selected_features:
			#feature_vector[fname] = 1           
			#num_of_w = text_nl.count(word)
			feature_vector[fname] = bidist.freq(word)
	
	tol_text = nltk.pos_tag(text_t)
	
	#unigram part-of-speech features
	fdist_pos = nltk.FreqDist(tag for (word, tag) in tol_text)
	for word, freq in fdist_pos.items(): 
		fname = "UNIPOS_" + word
		if selected_features == None or fname in selected_features:
            #feature_vector[fname] = 1            
			#num_of_w = text_nl.count(word)
			feature_vector[fname] = fdist_pos.freq(word)
		
	my_bigrams_pos = list(bigrams(tol_text))
	#to create a list of bigram's part-of-speech 
	bi_part_of_speach = []
	for item in my_bigrams_pos:
		a = (item[0][1], item[1][1])
		bi_part_of_speach.append(a)
	
	bidist_pos = nltk.FreqDist(bi_part_of_speach)
	for word, freq in bidist_pos.items(): 
		fname = "BIPOS_" + word[0] + "_" + word[1]
		if selected_features == None or fname in selected_features:
			feature_vector[fname] = bidist_pos.freq(word)
			
# Adds a simple LIWC derived feature
def add_liwc_features(text, feature_vector):
    liwc_scores = word_category_counter.score_text(text)
   
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]

    if positive_score > negative_score:
        feature_vector["liwc:positive"] = 1
    else:
        feature_vector["liwc:negative"] = 1
  
# Adds all our features and returns the vector
def features(review_text, review_words):
    feature_vector = {}

    uni_dist = nltk.FreqDist(review_words)
    
    my_bigrams = list(bigrams(review_words))
    bi_dist = nltk.FreqDist(my_bigrams)
    
    add_lexical_features(uni_dist, bi_dist, feature_vector, review_text)
    add_liwc_features(review_text, feature_vector)
    return feature_vector
    
    
def evaluate(classifier, features_category_tuples, reference_text, data_set_name, output_file):
    accuracy_results_file = open(output_file, 'w', encoding='utf-8')
    accuracy_results_file.write('Results of {}:\n\n'.format(data_set_name))
    # test on the data
    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)     
    accuracy_results_file.write("{0:10s} {1:8.5f}\n\n".format("Accuracy", accuracy))
    
    features_only = []
    reference_labels = []
    for feature_vectors, category in features_category_tuples:
        features_only.append(feature_vectors)
        reference_labels.append(category)
        
    predicted_labels = classifier.classify_many(features_only)
    
    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)

    accuracy_results_file.write(str(confusion_matrix))
    accuracy_results_file.write('\n\n')
    accuracy_results_file.close()
    
   
# python3 DT_baseline_test.py dt−classifier.pickle restaurant-development.data dev-results.txt
# python3 DT_baseline_test.py dt−classifier.pickle restaurant-testing.data test-results.txt
if __name__ == '__main__':
	classifier_model = sys.argv[1]
	reviews = sys.argv[2] 
	output = sys.argv[3]
	
	texts = process_reviews(reviews)
		        
    # Make sure we split the same way every time for the live coding
	random.seed(0)
    
    # Make sure to randomize the reviews first!
	random.shuffle(texts)
    
    # Convert the data into feature vectors
	featuresets = [
		(features(review_text, review_words), label) 
		for (review_text, review_words, label) in texts
	]
		
	import pickle
	f = open( classifier_model , "rb") 
	classifier = pickle.load(f) 
	f.close()
	
	evaluate(classifier, featuresets, texts, reviews, output)
	