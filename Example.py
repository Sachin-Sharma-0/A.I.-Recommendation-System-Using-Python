#To use the code, you can call the recommend_documents function with a list of tags and a list of keywords, like this:

tags = ['machine learning', 'data science']
keywords = ['supervised learning', 'classification']

recommendations = recommend_documents(tags, keywords)
print(recommendations)
#This will return a dataframe containing the titles and URLs of the recommended documents, sorted by their total similarity scores.
