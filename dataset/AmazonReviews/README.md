# Amazon Reviews Dataset

Dataset Url : https://www.kaggle.com/snap/amazon-fine-food-reviews

-   Data includes:
    -   Reviews from Oct 1999 - Oct 2012
    -   568,454 reviews
    -   256,059 users
    -   74,258 products
    -   260 users with > 50 reviews
-   Data format (each row): `Id, ProductId,UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text` (For our document classify task only 'Score' column serve as target, and 'Text' column server as model input.) 

-   Data Process:
    -   We only use the Score & Text column in our Document Classify Task. So we filtered other columns. 
    -   And we do a simple train_test split operation in here (https://www.kaggle.com/columbine/amazonreviews-data-processing)
-   **You should download train.csv, dev.csv, test.csv from the output of the [kaggle kernel](https://www.kaggle.com/columbine/amazonreviews-data-processing), instead of Using the original dataset.** **Then Move them into this folder.**

