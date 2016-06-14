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
	
	#unigram features
	for word, freq in fdist.items():
		fname = "UNI_" + word 
		if selected_features == None or fname in selected_features:        
			feature_vector[fname] = text_nl.count(word)
			
	#bigram features
	for word, freq in bidist.items(): # fdist = nltk.FreqDist(review_words)
		fname = "BIGRAM_" + word[0] + "_" + word[1]
		if selected_features == None or fname in selected_features:
			feature_vector[fname] = text_nl.count(word)
			
	
# Adds a simple LIWC derived feature
def add_liwc_features(text, feature_vector):
    liwc_scores = word_category_counter.score_text(text)
    
    #set 1 of liwc features  
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]

    if positive_score > negative_score:
        feature_vector["liwc:positive"] = 1
    else:
        feature_vector["liwc:negative"] = 1
        
    feature_vector["liwc:anger" + "_" + str(bin(round(liwc_scores["Anger"])))] = 1
    feature_vector["liwc:optimism"+ "_" + str(bin(round(liwc_scores["Optimism and energy"])))] = 1
    feature_vector["liwc:Swear_Words"+ "_" + str(bin(round(liwc_scores["Swear Words"])))] = 1
    feature_vector["liwc:sad"+ "_" + str(bin(round(liwc_scores["Sadness"])))] = 1
    
    #set 2 of liwc features
    feature_vector["liwc:Negations" +"_" + str(bin(round(liwc_scores["Negations"])))] = 1    
    feature_vector["liwc:Family"+ "_" + str(bin(round(liwc_scores["Family"])))] = 1
    feature_vector["liwc:Friends"+ "_" + str(bin(round(liwc_scores["Friends"])))] = 1
    feature_vector["liwc:Anxiety"+ "_" + str(bin(round(liwc_scores["Anxiety"])))] = 1
    feature_vector["liwc:Feel"+ "_" + str(bin(round(liwc_scores["Feel"])))] = 1
    feature_vector["liwc:Positive feelings"+ "_" + str(bin(round(liwc_scores["Positive feelings"])))] = 1

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
    
#python3 restaurant-competition-P2.py restaurant-competition-model-P2.pickle restaurant-testing.data "output-best.txt"
if __name__ == '__main__':
	classifier_model = sys.argv[1]
	reviews = sys.argv[2] 
	output = sys.argv[3]
	
	test = process_reviews(reviews)
	random.seed(0)
	random.shuffle(test)
	
	featuresets = [
        (features(review_text, review_words), label) 
        for (review_text, review_words, label) in test]
        
	import pickle
	f = open( classifier_model , "rb") 
	classifier = pickle.load(f) 
	f.close()
    
	evaluate(classifier, featuresets, test, "restaurant-testing.data", output)  