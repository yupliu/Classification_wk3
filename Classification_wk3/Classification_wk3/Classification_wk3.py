import graphlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn
except ImportError:
    pass

loans = graphlab.SFrame('C:\\Machine_Learning\\Classification_wk3\\lending-club-data.gl\\')
loans['grade'].show()
loans['home_ownership'].show()
loans['safe_loans'] = loans['bad_loans'].apply(lambda x:+1 if x==0 else -1)
loans = loans.remove_column['bad_loans']
loans['safe_loans'].show(view='Categorical')
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)
print "Percentage of safe loans  :", float(len(safe_loans_raw))/(len(safe_loans_raw)+len(risky_loans_raw))
print "Percentage of risky loans :", float(len(risky_loans_raw))/(len(safe_loans_raw)+len(risky_loans_raw))
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
risky_loans = risky_loans_raw
#undersample
safe_loans = safe_loans_raw.sample(percentage,seed=1)
# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

train_data, validation_data = loans_data.random_split(.8, seed=1)
decision_tree_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                target = target, features = features)
small_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 2)

small_model.show(view="Tree")

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

sample_validation_pred = decision_tree_model.predict(sample_validation_data)
print(sample_validation_pred)
correct = sample_validation_pred == sample_validation_data['safe_loans']
print(correct.sum())
sample_validation_prob = decision_tree_model.predict(sample_validation_data,output_type='probability')
print(sample_validation_prob)

small_validation_prob = small_model.predict(sample_validation_data,output_type='probability')
print(small_validation_prob)
