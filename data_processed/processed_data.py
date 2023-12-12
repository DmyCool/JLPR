import os
import torch
import numpy as np
import pandas as pd
from create_tract import create_tract
import geopandas as gpd
import math
import shutil
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


def process_trip():
    trip = pd.read_csv('./data/trip/Taxi_Trips_from_2013to202210.csv')
    trip['year'] = pd.to_datetime(trip['Trip Start Timestamp']).dt.year  # extract year
    trip = trip[trip['year'] < 2019]
    trip = trip[['Pickup Census Tract', 'Dropoff Census Tract', 'Pickup Community Area','Dropoff Community Area',
                 'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']]
    trip = trip.dropna(axis=0, how='any')

    trip.to_csv('./data/trip/2013_2018_taxi_trip_clean.csv')

def create_flow_mtx():
    trip = pd.read_csv('./data/trip/2013_2018_taxi_trip_clean.csv')

    pickup_tract = set(trip['Pickup Census Tract'])
    dropoff_tract = set(trip['Dropoff Census Tract'])
    _, tractid_number = create_tract()

    pickup_tract_ = list(pickup_tract - set(tractid_number))
    dropoff_tract_ = list(dropoff_tract - set(tractid_number))

    if pickup_tract_ != [] or dropoff_tract_ != []:
        trip = trip[-((trip['Pickup Census Tract'] == pickup_tract_) | (trip['Dropoff Census Tract'] == dropoff_tract_))]

    trip = trip[-(trip['Pickup Census Tract'] == trip['Dropoff Census Tract'])]

    trip = trip.astype(np.int64)
    trip['count'] = 0
    OD = trip.groupby(['Pickup Census Tract', 'Dropoff Census Tract'])['count'].count()



    flow_mtx = np.zeros((len(tractid_number), len(tractid_number)))
    for i in range(len(tractid_number)):
        for j in range(len(tractid_number)):
            if (tractid_number[i], tractid_number[j]) in OD.index:
                flow_mtx[i][j] = OD[tractid_number[i], tractid_number[j]]
            else:
                flow_mtx[i][j] = 0

    print(flow_mtx.shape)
    if not os.path.exists('./data/flow_mtx.npy'):
        np.save('./data/flow_mtx.npy', flow_mtx)

    return flow_mtx


def poi_features():
    poi = pd.read_csv('./data/poi/all_poi.csv', header=0, names=['VenueID', 'Lat', 'Lon', 'Venue_Cat', 'category'])
    poi = gpd.GeoDataFrame(poi, geometry=gpd.points_from_xy(poi['Lon'], poi['Lat']))
    poi = poi.set_crs(crs=4326)

    chicago, tractid_number = create_tract()
    join = gpd.sjoin(poi, chicago, how='inner', op='intersects')
    join['geoid10'] = join['geoid10'].astype(np.int64)

    join['count'] = 0
    poi_mtx = join.groupby(['geoid10', 'category'], as_index=False)['count'].count().pivot('geoid10', 'category', 'count')

    if len(poi_mtx) != len(tractid_number):
        missing_tract = [index for index in tractid_number if index not in list(poi_mtx.index)]
        missing_df = pd.DataFrame(index=missing_tract, columns=[col for col in poi_mtx])

        poi_mtx = poi_mtx.append(missing_df, ignore_index=False)

    poi_mtx.fillna(0, inplace=True)

    poi_mtx = poi_mtx.sort_index()
    poi_fre_mtx = poi_mtx.values
    print(poi_fre_mtx.shape)

    poi_tf_idf = []
    for index, row in poi_mtx.iterrows():
        tmp_tfidf = []
        for col in poi_mtx.columns.tolist():
            if sum(row) == 0.0:
                tf = row[col] / (sum(row) + 1)
            else:
                tf = row[col] / sum(row)

            idf = math.log(poi_mtx.shape[0] / len(poi_mtx[poi_mtx[col] != 0.0] + 1))

            tf_idf = tf * idf
            tmp_tfidf.append(tf_idf)

        poi_tf_idf.append(tmp_tfidf)

    poi_tfidf_mtx = np.array(poi_tf_idf)
    if not os.path.exists('./data/poi_fre_mtx.npy'):
        np.save('./data/poi_fre_mtx.npy', poi_fre_mtx)

    return poi_fre_mtx, poi_tfidf_mtx


def sample_image():
    df = pd.read_csv('./data/image/imgID_tractID.csv')
    geoid = set(df['geoid10'])
    sample_file = pd.DataFrame()
    for id in geoid:
        sample_condition = df[df['geoid10'] == id]
        if len(sample_condition) < 50:
            subset = sample_condition.sample(n=len(sample_condition))
            print('samples numbers less than the input sample numbers', id, len(sample_condition))
        else:
            subset = sample_condition.sample(n=50)
        sample_file = sample_file.append(subset)

    if not os.path.exists('./data/image/image_sample_index.csv'):
        sample_file.to_csv('./data/image/image_sample_index.csv', index=False)



def get_img():
    dir_path = './data/Panorama/'
    sample_index = pd.read_csv('./data/image/image_sample_index.csv')
    sample_id = list(sample_index['imgID'])

    for filename in os.listdir(dir_path):
        picpath = dir_path + filename + '\\Current\\'
        for picname in os.listdir(picpath):
            if picname.endswith('.jpg'):
                pointid = int(picname.split('_')[0])

                if pointid in sample_id:
                    old_file = picpath + picname
                    new_file = './data/image/SVI/' + picname
                    shutil.copyfile(old_file, new_file)

def feature_extract(imgpath):
    im = Image.open(imgpath)

    model_extractor = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')

    model_extractor.fc = nn.Linear(2048, 128)
    model_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(im)
    img.unsqueeze_(dim=0)
    feat = model_extractor(img)
    feat = feat.detach().numpy()
    return feat


def get_image_features(filepath, indexfile):
    _, tractid_numbers = create_tract()
    train_img = pd.read_csv(indexfile)

    img_features = []
    for i in range(len(tractid_numbers)):
        print('processing current tract:', tractid_numbers[i])
        temp = train_img.loc[train_img['geoid10'] == tractid_numbers[i]]
        imgid = list(temp['imgID'])

        feat_list = []
        for imgfile in os.listdir(filepath):
            pointid = int(imgfile.split('_')[0])
            if pointid in imgid:
                imgpath = filepath + imgfile
                feat = feature_extract(imgpath)
                feat_list.append(feat)


        mean_feat = np.mean(np.array(feat_list), axis=0)  # (1, 128)
        img_features.append(mean_feat)


    if len(img_features.shape) == 3:
        img_features = img_features.squeeze()

    print(img_features.shape)
    if not os.path.exists('./data/img_feat_mtx.npy'):
        np.save('./data/img_feat_mtx.npy', img_features)

    return img_features



def crime_process():
    crime2018 = pd.read_csv('./data/crime/Crime2018.csv', encoding='utf-8')
    crime2019 = pd.read_csv('./data/crime/Crime2019.csv', encoding='utf-8')
    train = crime2018.dropna(axis=0, how='any')
    test = crime2019.dropna(axis=0, how='any')
    # crime2010 = pd.read_csv('../data/Crimes_-_2010.csv', encoding='utf-8')
    # crime2011 = pd.read_csv('../data/Crimes_-_2011.csv', encoding='utf-8')
    # train = crime2010.dropna(axis=0, how='any')
    # test = crime2011.dropna(axis=0, how='any')

    crime_train = gpd.GeoDataFrame(train, geometry=gpd.points_from_xy(train['Longitude'], train['Latitude'], crs="EPSG:4326"))
    chicago, tractid_number = create_tract()
    train_sjoin = gpd.sjoin(crime_train, chicago, how='inner', op='intersects')
    train_sjoin['geoid10'] = train_sjoin['geoid10'].astype(np.int64)

    crime_test = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(test['Longitude'], test['Latitude'], crs="EPSG:4326"))
    chicago, tractid_number = create_tract()
    test_sjoin = gpd.sjoin(crime_test, chicago, how='inner', op='intersects')
    test_sjoin['geoid10'] = test_sjoin['geoid10'].astype(np.int64)

    print(len(train_sjoin), len(test_sjoin))

    return train_sjoin, test_sjoin

def get_crime_count(train, test):
    _, tractid_number = create_tract()

    train['count'] = 0
    test['count'] = 0
    train_crime_count = train.groupby(['geoid10'])['count'].count()
    test_crime_count = test.groupby(['geoid10'])['count'].count()

    if len(train_crime_count) != len(tractid_number):
        missing_tract = [index for index in tractid_number if index not in list(train_crime_count.index)]
        missing_series = pd.Series(index=missing_tract)
        train_crime_count = train_crime_count.append(missing_series, ignore_index=False)

    if len(test_crime_count) != len(tractid_number):
        missing_tract = [index for index in tractid_number if index not in list(test_crime_count.index)]
        missing_series = pd.Series(index=missing_tract)
        test_crime_count = test_crime_count.append(missing_series, ignore_index=False)

    train_crime_count.fillna(0, inplace=True)
    test_crime_count.fillna(0, inplace=True)

    train_crime_count = train_crime_count.sort_index()
    test_crime_count = test_crime_count.sort_index()

    print(train_crime_count.shape, test_crime_count.shape)

    if not os.path.exists('./data/crime_train.npy'):
        np.save('../data/crime_train2010.npy', train_crime_count)
        np.save('../data/crime_test2011.npy', test_crime_count)



def poi_encoding():
    poi_fre_mtx = torch.tensor(np.load('../data/poi_fre_mtx.npy'), dtype=torch.float)
    img_feat_mtx = torch.tensor(np.load('../data/img_feat_mtx.npy'), dtype=torch.float)

    poi_encoding_layer = nn.Linear(in_features=14, out_features=16)
    img_encoding_layer = nn.Linear(in_features=128, out_features=128)

    poi_embeddings = poi_encoding_layer(poi_fre_mtx)
    img_embeddings = img_encoding_layer(img_feat_mtx)


    np.save('../data/poi_emb_mtx.npy', poi_embeddings.detach().numpy())
    np.save('../data/img_emb_mtx.npy', img_embeddings.detach().numpy())



if __name__ == '__main__':
    # flow_mtx = create_flow_mtx()
    # poi_mtx = poi_features()

    # sample_image()
    # get_image_features('./data/image/SVI/', './data/image/image_sample_index.csv')

    train, test = crime_process()
    get_crime_count(train, test)

    # poi_encoding()
