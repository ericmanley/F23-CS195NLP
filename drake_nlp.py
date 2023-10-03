from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import random
from pprint import pformat

class MarkovModel:
    
    def __init__(self):
        # empty nested dictionary mapping words to words to ints
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        
    def train(self,corpus):
        #loop through each word in the corpus record the next word
        #in its frequency dictionary
        for idx in range(len(corpus)-1):
            current_token = corpus[idx]
            next_token = corpus[idx+1]
            self.transition_counts[current_token][next_token] += 1
            
    def generate_random_next_word(self,current_word):
        #get the frequency of all words that come after current_word
        possible_words_counts = self.transition_counts[current_word]
        #count up the total of all words that come after current_word
        total_occurrences = sum(possible_words_counts.values())

        #we are going to select one occurence randomly
        random_num = random.randint(1,total_occurrences)

        #subtract words counts from our random number until we hit 0
        #this will hit more frequent words proportionally more often
        for word in possible_words_counts:
            random_num = random_num - possible_words_counts[word]
            if random_num <= 0:
                return word
            
    def next_state_probabilities(self,current_state):
        #get the frequency of all states that come after current_word
        possible_state_counts = self.transition_counts[current_state]
        #count up the total of all words that come after current_word
        total_occurrences = sum(possible_state_counts.values())

        state_probabilities = defaultdict(float)
        for word in possible_state_counts:
            state_probabilities[word] = possible_state_counts[word]/total_occurrences
        
        return state_probabilities
    
    def get_state_list(self):
        return list(self.transition_counts.keys())

           
    def generate_text(self,num=100,start_word = "I"):
        #a running string to build on with random words
        markov_text = start_word + " "
        curr_word = start_word
        
        #add num random words onto our running string
        for n in range(num):
            curr_word = self.generate_random_next_word(curr_word)
            markov_text += curr_word 
            markov_text += " "
            
        return markov_text
    
    def __str__(self):
        # convert defaultdicts to dicts and format using the pprint formatter
        return pformat({key:dict(self.transition_counts[key]) for key in self.transition_counts})
    
    def visualize(self,probabilities=False,layout=nx.kamada_kawai_layout):
        # use this method to generate visualizations of small models
        # it will take too long on large texts - don't do it!
        G = nx.DiGraph()
        
        if probabilities:
            transition_probabilities = defaultdict(dict)
            for current_word, next_words in self.transition_counts.items():
                total_occurrences = sum(next_words.values())
                for next_word, count in next_words.items():
                    transition_probabilities[current_word][next_word] = count / total_occurrences
                    
            for current_word, next_words in transition_probabilities.items():
                for next_word, probability in next_words.items():
                    G.add_edge(current_word, next_word, weight=probability)
        else:
            for current_word, next_words in self.transition_counts.items():
                for next_word, count in next_words.items():
                    G.add_edge(current_word, next_word, weight=count)

        pos = layout(G)
        edge_labels = {
            edge: f"{G.edges[edge]['weight']:.2f}" if G.edges[edge]['weight'] % 1 != 0 else f"{G.edges[edge]['weight']:.1f}"
            for edge in G.edges()
        }
        #edge_labels = {edge: f"{G.edges[edge]['weight']}" for edge in G.edges()}
        #nx.draw_networkx_edges(G , node_pos,)
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10,connectionstyle='arc3, rad = 0.1')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,label_pos=0.44)

        plt.title("Markov Model Visualization")
        plt.show()