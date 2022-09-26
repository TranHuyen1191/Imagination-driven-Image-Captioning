import pandas as pd 

## Extract imagination-driven captions
# List of keywords
keywords = {'looks like','look like','look as','looks as','reminds me','remind me',
            'is like','is likely','are like','are likely','think of','thinks of',
            'as if','as though','feel like','feels like','shaped like', 'shapes like', 'shape like',
            'calm like','looks likely','look likely',
            'seems like','seem like','seems as', 'seem as',
            'looks almost like','look almost like','is almost as','are almost as','seems to be', 'seem to be',
            'resemble','resembling'}

def count_IdC(df):
    IdC_df = pd.DataFrame(columns= df.columns) 
    LC_df = pd.DataFrame(columns= df.columns) 
    for _,row in df.iterrows():
        candidate = row.caption
        idcflag =  any([kw in candidate for kw in keywords])
        if idcflag:
            IdC_df = IdC_df.append(row,ignore_index=True)
        else:
            LC_df = LC_df.append(row,ignore_index=True)
    return IdC_df,LC_df