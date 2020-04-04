from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
    def __init__( self, filename ):
        self.filename = filename
        self.word2index = {}
        self.word2count = {}
        self.sentences = []
        self.index2word = { 0: "SOS", 1: "EOS" }
        self.n_words = 2  # Count SOS and EOS

        with open( self.filename ) as fd:
            for i, line in enumerate( fd.readlines() ):
                line = line.strip()
                self.sentences.append( line )
        self.allow_list = [ True ] * len( self.sentences )
        self.target_sentences = self.sentences[ :: ]
                
    def get_sentences( self ):
        return self.sentences[ :: ] 

    def get_sentence( self, index ):
        return self.sentences[ index ]

    def choice( self ):
        while True:
            index = random.randint( 0, len( self.allow_list ) - 1 )
            if self.allow_list[ index ]:
                break
        return self.sentences[ index ], index

    def get_allow_list( self, max_length ):
        allow_list = []
        for sentence in self.sentences:
            if len( sentence.split() ) < max_length:
                allow_list.append( True )
            else:
                allow_list.append( False )
        return allow_list
                    
    def load_file( self, allow_list = [] ):
        if allow_list:
            self.allow_list = [x and y for (x,y) in zip( self.allow_list, allow_list ) ]
        self.target_sentences = []
        for i, sentence in enumerate( self.sentences ):
            if self.allow_list[ i ]:
                self.addSentence( sentence )
                self.target_sentences.append( sentence )
                    
    def addSentence( self, sentence ):
        for word in sentence.split():
            self.addWord(word)
            

    def addWord( self, word ):
        if word not in self.word2index:
            self.word2index[ word ] = self.n_words
            self.word2count[ word ] = 1
            self.index2word[ self.n_words ] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1        

# Start core part
class Encoder( nn.Module ):
    def __init__( self, input_size, hidden_size ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding( input_size, hidden_size )
        self.gru         = nn.GRU( hidden_size, hidden_size )

    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size ).to( device )

    def forward( self, _input, hidden ):
        embedded = self.embedding( _input ).view( 1, 1, -1 )
        out, new_hidden = self.gru( embedded, hidden )
        return out, new_hidden

class Decoder( nn.Module ):
    def __init__( self, hidden_size, output_size ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding( output_size, hidden_size )
        self.gru         = nn.GRU( hidden_size, hidden_size )
        self.out         = nn.Linear( hidden_size, output_size )
        self.softmax     = nn.LogSoftmax( dim = 1 )
        
    def forward( self, _input, hidden ):
        output = self.embedding( _input ).view( 1, 1, -1 )
        output = F.relu( output )
        output, hidden = self.gru( output, hidden )
        output = self.softmax( self.out( output[ 0 ] ) )
        return output, hidden
    
    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size ).to( device )

def tensorFromSentence( lang, sentence ):
    indexes = [ lang.word2index[ word ] for word in sentence.split(' ') ]
    indexes.append( EOS_token )
    return torch.tensor( indexes, dtype=torch.long ).to( device ).view(-1, 1)

def tensorsFromPair( input_lang, output_lang ):
    input_sentence, index = input_lang.choice()
    output_sentence = output_lang.get_sentence( index )
    
    input_tensor = tensorFromSentence( input_lang, input_sentence )
    output_tensor = tensorFromSentence( output_lang, output_sentence )
    return (input_tensor, output_tensor)

def main():
    n_iters       = 75000
    learning_rate = 0.01
    hidden_size   = 256
    max_length = 10
    input_lang = Lang( 'fr.txt' )
    output_lang = Lang( 'en.txt')
    
    allow_list = [x and y for (x,y) in zip( input_lang.get_allow_list( max_length ), output_lang.get_allow_list( max_length ) ) ]
            
    input_lang.load_file( allow_list )
    output_lang.load_file( allow_list )
    
    
    encoder = Encoder( input_lang.n_words, hidden_size ).to( device )
    decoder = Decoder( hidden_size, output_lang.n_words ).to( device )
    
    encoder_optimizer = optim.SGD( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.SGD( decoder.parameters(), lr=learning_rate )

    training_pairs = [ tensorsFromPair( input_lang, output_lang ) for i in range( n_iters ) ]
    criterion = nn.NLLLoss()
    
    for epoch in range( 1, n_iters + 1):
        
        input_tensor, output_tensor = training_pairs[ epoch - 1 ]

        encoder_hidden = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
        input_length  = input_tensor.size(0)
        output_length = output_tensor.size(0)
        
        # Encoder phese
        for i in range( input_length ):
            encoder_output, encoder_hidden = encoder( input_tensor[ i ], encoder_hidden )
            
        # Decoder phese
        loss = 0
        decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
        decoder_hidden = encoder_hidden
        for i in range( output_length ):
            decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
            decoder_input = output_tensor[ i ]
            if random.random() < 0.5: 
                topv, topi                     = decoder_output.topk( 1 )
                decoder_input                  = topi.squeeze().detach() # detach from history as input
            loss += criterion( decoder_output, output_tensor[ i ] ) 
            if decoder_input.item() == EOS_token: break
            
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        if epoch % 10 == 0:
            print( "[epoch num %d (%d)] [ loss: %f]" % ( epoch, n_iters, loss.item() / output_length ) )
                   

if __name__ == '__main__':
    main()
    
