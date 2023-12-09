train_df = pd.read_csv('train.csv', index_col='id')
train_df.head()

plt.figure(figsize=(12,6))
plt.title("Target value Distributions")
sns.distplot(train_df['target'], kde=True, hist=False, bins=240, label='target')
plt.show()

temp = train_df['target'].apply(lambda x: "non-toxic" if x < 0.5 else "toxic")

fig, ax = plt.subplots(1,1,figsize=(5,5))
total = float(len(temp))
cntplot = sns.countplot(temp)
cntplot.set_title('Percentage of non-toxic and toxic comments')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.0, height + 3, '{:1.2f}%'.format(100*height/total), ha='center')
    
plt.show()
