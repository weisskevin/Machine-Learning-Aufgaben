import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Lese den Datensatz
df = pd.read_fwf('../datensaetze/artest.csv', widths=[1] * 50, header=None)
# Benenne die Spalten entsprechend der Produkte

column_names = ['P'+ str(i+1) for i in range(50)]
row_names = ['Kassenbon' + str(i+1) for i in range(10000)]

#df.columns = column_names
#df.index = row_names

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

print("Assoziationsregeln:")
print(rules)
