import glob
import os
import random
from math import log
from PIL import Image, ImageOps

def get_image_files( folder_path, picnum ):
    return random.sample( glob.glob( os.path.join( folder_path, '*.jpg' ) ), picnum )


def main():
    female = get_image_files( './female', 50 )
    male = get_image_files( './male', 50 )
    crop_size =32

    # data structure
    # [ [ ('F', [....] ), ('M', [....]), ('F', [....]) ], [ ('F', [....] ), ('M', [....]), ('F', [....])] ]
    sets = []
    
    sets += [ ('F', get_pixels( path, crop_size, crop_size ) ) for path in female ]
    sets += [ ('M', get_pixels( path, crop_size, crop_size ) ) for path in male ]

    size = 32 * 32
    
    train( sets, size )

def get_pixels( path, w, h, crop_area = 0.86 ):
    img = Image.open( path )
    width, height = img.size
    margin_ratio = (1-crop_area)/2.0
    sx = (width * margin_ratio)
    sy = (height * margin_ratio)
    
    img = img.convert( 'L' )
    img = ImageOps.equalize(img) # need?
    img = img.crop((sx, sy, width * crop_area, height*crop_area)).resize( (w, h), Image.BICUBIC )
    return list( img.getdata() )
    
# [ [], [] ]
def entropy( sets ):
    log2 = lambda x: log(x) / log(2)
    
    def get_histogram( keys, sets ):
        hist = dict( zip( keys, [0]*len( keys ) ) )
        for s in sets:
            k, _ = s
            hist[ k ] += 1
        return hist

    unique_keys = set( [ _data[ 0 ] for _data in sets] )
    histogram = get_histogram( unique_keys, sets )
    
    ent = 0.0
    for k in unique_keys:
        p = histogram[ k ] / len( sets ) 
        if p != 0:
            ent = ent - p * log2( p )
    return ent

def split( sets, idx, value ):
    set1 = []
    set2 = []
    for data in sets:
        label, pixels = data
        if pixels[ idx ] >= value:
            set1.append( data )
        else:
            set2.append( data )
    return [set1, set2]

def train( sets, size ):
    if len( sets ) == 0: return 0
    step = 2
    current_score = entropy( sets )
    
    best_gain     = 0
    best_criteria = None
    best_sets     = None
    
    for index in range( 0, size, step ):
        for value in range( 256 ):
            set1, set2 = split( sets, index, value )
            p = float( len( set1 ) ) / len( sets )
            
            gain = current_score - p * entropy( set1 ) - ( 1 - p ) * entropy( set2 )
            if gain > best_gain and len( set1 ) > 0 and len( set2 ) > 0:
                best_gain = gain
                best_criteria =( index, value )
                best_sets=( set1, set2 )
    if best_gain > 0:
        true_branch = train( best_sets[ 0 ], size )
        flase_branch = train( best_sets[ 1 ], size )
        
if __name__ == '__main__':
    main()
