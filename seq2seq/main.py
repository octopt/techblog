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

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

class Encoder( nn.Module ):
    def __init__( self, input_size, hidden_size ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding( input_size, hidden_size )
        self.gru         = nn.GRU( hidden_size, hidden_size )

    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size )

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
        return torch.zeros( 1, 1, self.hidden_size )
    
teacher_forcing_ratio = 0.5


# def train( input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):


def tensorFromSentence( lang, sentence ):
    indexes = [ lang.word2index[ word ] for word in sentence.split(' ') ]
    indexes.append( EOS_token )
    return torch.tensor( indexes, dtype=torch.long ).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence( input_lang, pair[ 0 ])
    output_tensor = tensorFromSentence( output_lang, pair[ 1 ] )
    return (input_tensor, output_tensor)

def main():
    n_iters = 75000
    learning_rate=0.01
    hidden_size = 256
    # MAX_LENGTH = 10
    
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    
    encoder = Encoder( input_lang.n_words, hidden_size )
    decoder = Decoder( hidden_size, output_lang.n_words )
    
    encoder_optimizer = optim.SGD( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.SGD( decoder.parameters(), lr=learning_rate )
    
    training_pairs = [ tensorsFromPair( random.choice( pairs ), input_lang, output_lang ) for i in range( n_iters ) ]
                       
    criterion = nn.NLLLoss()
    
    for i in range( 1, n_iters + 1):
        
        input_tensor, output_tensor = training_pairs[ i - 1 ]

        encoder_hidden = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
        input_length  = input_tensor.size(0)
        output_length = output_tensor.size(0)

        for ei in range( input_length ):
            encoder_output, encoder_hidden = encoder( input_tensor[ ei ], encoder_hidden )
            
        loss = 0
        
        decoder_input  = torch.tensor( [ [ SOS_token ] ] )
        decoder_hidden = encoder_hidden

        
        for di in range( output_length ):
            decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
            decoder_input = output_tensor[ di ]
            if random.random() < 0.5: 
                topv, topi                     = decoder_output.topk( 1 )
                decoder_input                  = topi.squeeze().detach() # detach from history as input
            loss += criterion( decoder_output, output_tensor[ di ] ) 
            if decoder_input.item() == EOS_token: break
            
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        print( loss.item() / output_length )

class Lang:
    def __init__( self, name ):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = { 0: "SOS", 1: "EOS" }
        self.n_words = 2  # Count SOS and EOS

    def addSentence( self, sentence ):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord( self, word ):
        if word not in self.word2index:
            self.word2index[ word ] = self.n_words
            self.word2count[ word ] = 1
            self.index2word[ self.n_words ] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1        

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )        

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s        
        
def readLangs(lang1, lang2, reverse=False):
    # Read the file and split into lines
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [ [ normalizeString(s) for s in l.split('\t') ] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs( lang1, lang2, reverse )
    pairs = filterPairs( pairs )
    
    for pair in pairs:
        input_lang.addSentence( pair[0] )
        output_lang.addSentence( pair[1] )
    print( "Counted words:" )
    print( input_lang.name, input_lang.n_words )
    print( output_lang.name, output_lang.n_words )
    return input_lang, output_lang, pairs

def filterPair( p ):
    eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s ", "you are", "you re ", "we are", "we re ", "they are", "they re " )
    return ( len(p[0].split(' ')) < MAX_LENGTH ) and ( len(p[1].split(' ')) < MAX_LENGTH  ) and p[1].startswith( eng_prefixes )

def filterPairs(pairs):
    return [ pair for pair in pairs if filterPair( pair )] 

if __name__ == '__main__':
    main()
    # input_lang, output_lang, pairs = prepareData( 'eng', 'fra', True )
    # print( random.choice( pairs ) )
    
