import torch

from main import Lang
from main import Decoder
from main import Encoder
from main import tensorFromSentence

SOS_token = 0
EOS_token = 1
device = 'cuda'

def evaluate( sentence, max_length ):
    input_lang  = Lang( 'jpn.txt')
    output_lang = Lang( 'eng.txt' )
    
    allow_list  = output_lang.get_allow_list( max_length )

    input_lang.load_file( allow_list )
    output_lang.load_file( allow_list )
        
    hidden_size = 256
    embedding_size =256 
    encoder = Encoder( input_lang.n_words, embedding_size, hidden_size ).to( device )
    decoder = Decoder( hidden_size, embedding_size, output_lang.n_words ).to( device )


    enfile = "encoder_75000_5.399948"
    defile = "decoder_75000_5.399948"
    encoder.load_state_dict( torch.load( enfile ) )
    decoder.load_state_dict( torch.load( defile ) )
    
    with torch.no_grad():
        input_tensor   = tensorFromSentence(input_lang, sentence)
        input_length   = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input      = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden     = encoder_hidden
        decoded_words      = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
                
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]    

if __name__ == '__main__':
    import MeCab
    import unicodedata
    wakati = MeCab.Tagger("-Owakati")
    sentence = 'とても嬉しいです.'
    sentence = unicodedata.normalize( "NFKC", sentence.strip() )
    a=wakati.parse( sentence.strip() ).split()
    ret =" ".join( a )
    print( evaluate( ret, 30 ) )
