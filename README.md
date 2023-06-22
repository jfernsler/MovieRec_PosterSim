## Movie Recommender with Poster image similarity

This is a movie recommender neural net model trained on the MovieLens 1M Dataset. It will make a recommendation of 15 potential movies to a user and then further filter that set by doing a similarity check of highly rated movies against the potential recommened movies. 

## Pitch
While the purpose of a recommender system for streaming services such as Netflix, Prime Video, Hulu, and others is to provide options that the viewer may enjoy based on several metrics, the real key is to get them to begin to watch their next video as soon as possible. Because we all browse these options visually with video key art, I propose an ensemble of models to hone in on a collection of recommended movies, and then fine tune that selection by finding the key art that is most similar to their highest rated movies. The concept being that they may be drawn to a visually similar recommended movie more-so than one which has a wildly different poster.

## Data source
While the main source of my data is the MovieLens 1M Dataset from grouplens.org, I further augmented it by appending imdbID values pulled from the MovieLens 25M Dataset. This was difficult to due to the very different tilting and id values between the two. Initially I was looking for a dataset of posters as well, but included a poster scraper to pull down the posters as needed from IMDB.com.

## Model and Data Justification
After some research, I used a deep neural network with sparse embeddings for each of the values provided. For users I used gender (2 values), ages (7 values), occupation (21 values) and for movies I had access to genres which provided just over 320 combinations. By using this method we can attempt to avoid cold start issues as we have user data driving the overall movie selection - with richer data available about the user and movies, the movie predictions can become even more directed. For poster similarity, I take the top three rated movies from the user and find the cluster center using embedding values from the ResNet18 model, then compare the 30 top predictions to that center using their embedding values and select the top 5 results from that.
      
## Commented Examples
The recommendations are sometimes skewed towards highly rated movies and transcend the genres a bit more than I’d like. However the poster similarity does bring in some interesting choices to the round, and certainly has a visual appeal. What follows are two predictions made just from the model, and then two more predictions made using the image similarity filtering. 

One thing to notice in the first two examples is the repetition of recommended movies. There's a definate weight of overall highly rated movies. This is where the additional use of poster similarity helps out.

### Model only predictions
******************************
Top 3 rated movies for user 3355
1. Title: Ghost in the Shell (Kokaku kidotai) (1995) :: Rating: 5.0 
2. Title: Malcolm X (1992) :: Rating: 5.0
3. Title: One Flew Over the Cuckoo's Nest (1975) :: Rating: 5.0

Top 10 _predicted_ movies for user 3355
1. Title: Atlantic City (1980) :: Predicted Rating: 4.965
2. Title: JFK (1991) :: Predicted Rating: 4.940
3. Title: Saturn 3 (1979) :: Predicted Rating: 4.906
4. Title: Quatermass II (1957) :: Predicted Rating: 4.901
5. Title: Cimarron (1931) :: Predicted Rating: 4.888
6. Title: War of the Worlds, The (1953) :: Predicted Rating: 4.877
7. Title: Runaway (1984) :: Predicted Rating: 4.873
8. Title: Stalker (1979) :: Predicted Rating: 4.868
9. Title: Mister Roberts (1955) :: Predicted Rating: 4.861
10. Title: Sting, The (1973) :: Predicted Rating: 4.861

******************************
Top 3 rated movies for user 6003
1. Title: Doctor Zhivago (1965) :: Rating: 5.0 
2. Title: Wild Bunch, The (1969) :: Rating: 5.0 
3. Title: Little Big Man (1970) :: Rating: 5.0

Top 10 _predicted_ movies for user 6003
1. Title: Saturn 3 (1979) :: Predicted Rating: 4.967
2. Title: Quatermass II (1957) :: Predicted Rating: 4.962
3. Title: Cimarron (1931) :: Predicted Rating: 4.949
4. Title: War of the Worlds, The (1953) :: Predicted Rating: 4.938 5) Title: Runaway (1984) :: Predicted Rating: 4.934
6. Title: Stalker (1979) :: Predicted Rating: 4.929
7. Title: Mister Roberts (1955) :: Predicted Rating: 4.922
8. Title: Good Morning, Vietnam (1987) :: Predicted Rating: 4.920 9) Title: High Noon (1952) :: Predicted Rating: 4.917
10. Title: Species II (1998) :: Predicted Rating: 4.899

 ### Model plus poster similarity
 Here we get much more variety in the recoomendations.


******************************

![](/charts/sim_predictions_04.png)

Top 3 rated movies for user 3997
1. Title: Licence to Kill (1989) :: Rating: 5.0 
2. Title: Live and Let Die (1973) :: Rating: 5.0 
3. Title: Thunderball (1965) :: Rating: 5.0 

Top 5 predicted movies for user 3997
1. Title: In the Heat of the Night (1967) :: Predicted Rating: 4.8 :: Poster Similarity: 87.14% 
2. Title: Some Like It Hot (1959) :: Predicted Rating: 4.8 :: Poster Similarity: 86.11% 
3. Title: Cimarron (1931) :: Predicted Rating: 4.9 :: Poster Similarity: 84.31% 
4. Title: Runaway (1984) :: Predicted Rating: 4.9 :: Poster Similarity: 83.88% 
5. Title: High Noon (1952) :: Predicted Rating: 4.9 :: Poster Similarity: 83.85% 


******************************

![](/charts/sim_predictions_02.png)

Top 3 rated movies for user 3174
1. Title: Toy Story (1995) :: Rating: 5.0
2. Title: Heat (1995) :: Rating: 5.0
3. Title: Sliding Doors (1998) :: Rating: 5.0

Top 5 _predicted_ movies for user 3174
1. Title: Army of Darkness (1993) :: Predicted Rating: 4.5 :: Poster Similarity: 84.58% 
2. Title: Hear My Song (1991) :: Predicted Rating: 4.5 :: Poster Similarity: 83.17%
3. Title: Runaway (1984) :: Predicted Rating: 4.5 :: Poster Similarity: 82.73%
4. Title: JFK (1991) :: Predicted Rating: 4.6 :: Poster Similarity: 82.71%
5. Title: Golden Child, The (1986) :: Predicted Rating: 4.5 :: Poster Similarity: 82.16%

## Testing
I created a confusion matrix from all one million ratings and the model’s predicted ratings for each. It’s got a heavy bias towards the north east and doesn’t do well rating movies with less than 3-4 stars (normalized 0-1.0 here).

![](/charts/movie_rating_confusion_matrix_all_data.png)

This is almost certainly due to the bias inherent in the MovieLens 1M dataset as shown by this rating distribution histogram.

![](/charts/movie_ratings.png)
  
## Code and Instructions to Run it
The code, model, and a reduced set of data can be cloned from: 
* https://github.com/jfernsler/reco_similar_posters
* Ensure you have all necessary packages in the requirements.txt
Once cloned the primary script to run is in /src/a2_main.py. Run this script with one of the following flags:
* a2_main.py -psim
    * --predict_similar
    * Make a prediction for a random user based on the trained model and poster
    similarity.
    * This will display and save the poster similarity image (to the charts directory)
    * This will also download all necessary posters to perform this check - any previously
downloaded posters will be directly accessed.
* a2_main.py -p
    * —predict
    * Make a prediction based on the trained model only.
* a2_main.py -d
    * —test_dataset
    * Print-out of a random sampling from the dataset.

## Addendum
* I mention a the avoidance of the cold start problem, but this is obviously still an issue when dealing with the poster similarity as it’s based on previous viewings. This could be solved in a variety of ways including creating clusters and begin to see where a user tends to go.
* The merged datasets to acquire the imdbIDs ended up with nearly 600 missing values due to mismatches in the entered data between the two sets. While I addressed quite a few of these by searching, copying, and pasting, I ran out of time and left many still blank. These are treated with a black poster image and essentially dropped from the similarities.