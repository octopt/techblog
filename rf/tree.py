import glob
import os
import random
from math import log
from PIL import Image, ImageOps

import uuid
import pickle

# 画像をランダムにサンプリング
def get_image_files( folder_path, picnum ):
    return random.sample( glob.glob( os.path.join( folder_path, '*.jpg' ) ), picnum )

def tree_result( node, data ):
    if 'leaf' in node: return node[ 'leaf' ]
    index = node[ 'index' ]
    value = node[ 'value' ]
    
    cond = data[ index ] >= value
    return tree_result( node[ cond ], data )

def validation():
    # with open('ret_9f36aa16223a4d09a33e7fecfb89e59e_size_1024_picknum_50_treenum_50.pkl', 'rb') as fd:
    # with open('ret_ab19e7042e5242f2aa95ae4d4ca7e7d3_size_1024_picknum_30_treenum_50.pkl', 'rb') as fd:
    with open('ret_126f956207384267851ccdeaf2962c59_size_1024_picknum_20_treenum_100.pkl', 'rb') as fd:
    
        forest = pickle.load(fd)
    # print( forest )
    num_pick = 100
    female = get_image_files( './female', num_pick )
    male = get_image_files( './male', num_pick )

    crop_size =32
    sets = []
    # データセット作成
    sets += [ ('F', get_pixels( path, crop_size, crop_size ) ) for path in female ]
    sets += [ ('M', get_pixels( path, crop_size, crop_size ) ) for path in male ]

    count = 0
    for d in sets:
        label, pixels = d
        ans = tree_result( forest[ 0 ], pixels )
        if label == ans:
            count += 1
    print( "true ratio --> ", count / (num_pick*2) )

# female, maleフォルダより50個のファイルを取得する。
def main():
    forest = []
    num_tree = 100
    for i in range( num_tree ):
        print( f"building tree ... {i+1}/{num_tree}")
        num_pick = 20
        female = get_image_files( './female', num_pick )
        male = get_image_files( './male', num_pick )
        
        crop_size =32
        sets = []
        # データセット作成
        sets += [ ('F', get_pixels( path, crop_size, crop_size ) ) for path in female ]
        sets += [ ('M', get_pixels( path, crop_size, crop_size ) ) for path in male ]

        size = 32 * 32
        ret = train( sets, size )
        forest.append( ret )
    pkl_name = f"ret_{uuid.uuid4().hex}_size_{size}_picknum_{num_pick}_treenum_{num_tree}.pkl"
    print( "dumped to --> ", pkl_name )
    with open( pkl_name, 'wb' ) as fd:
        pickle.dump( forest, fd )

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
        ans_index, ans_value = best_criteria
        true_branch = train( best_sets[ 0 ], size )
        false_branch = train( best_sets[ 1 ], size )
        return { True: true_branch, False: false_branch, 'index': ans_index, 'value': ans_value}
    else:
        return {'leaf': sets[0][0] }
        
if __name__ == '__main__':
    # main()
    validation()
