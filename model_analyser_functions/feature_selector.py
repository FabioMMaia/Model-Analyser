
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

class Feature_Selector():

    def __init__(self,df, target):
        assert isinstance(df,pd.DataFrame), 'df must be a pandas dataframe'
        assert target in df.columns, 'target must be in df'

        self.df=df
        self.target = target
        self.seed=0

    
    def boxplot_custom(self, dict_values, label):
        pd.DataFrame(dict_values).plot.box(
            rot=90, figsize=(11,6), 
            medianprops=dict(linestyle='-', linewidth=1.8), 
            patch_artist=True)
        # plt.ylim([-0.05,1.05]); 
        plt.ylabel(label)

    def classifier_univar_test(self, model_params=None, model=None , plot=False):
        # assert isinstance(model_params,dict), 'model_params must be a dictionary'

        if model is None:
            model =  LogisticRegression(random_state=self.seed)

        accuracies={}

        for column in self.df.columns:
            if column!= self.target:
                
                acc_col_list=[]     

                CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

                X = self.df[[column]]
                y = self.df[self.target]

                for i,(id_train,id_test) in enumerate(CV.split(X,y)):

                    # return y.iloc[id_train]

                    model.fit(X.iloc[id_train],y.iloc[id_train])
                    y_true = y.iloc[id_test]
                    y_pred = model.predict(X.iloc[id_test])
                    acc = accuracy_score(y_true, y_pred)
                    acc_col_list.append(acc)
                
                accuracies[column] = acc_col_list

        if(plot):
            self.boxplot_custom(accuracies, 'Accuracies')

        return accuracies

    def create_fi_df(self,importance,names):
        
        #Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        fi_df.reset_index(inplace=True, drop=True)

        return fi_df


    def plot_feature_importance(self,fi_df, ax, title):

        # fi_df = fi_df.head(20)

        #Define size of bar plot
        # plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'],orient = 'h', ax=ax)
        #Add chart labels
        ax.set_title(title)
        # plt.title(model_type + ' | Feature Importance')
        # plt.xlabel('Feature Importance')
        # plt.ylabel('Names')
        # plt.show()

    def RandomForest_tester(self, model_params=None, plot=False):

        model_RF = RandomForestClassifier()

        CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        X = self.df.copy()
        X = X.drop(columns='target')
        X = X.assign(rd_var1 = np.random.random(X.shape[0]))
        X = X.assign(rd_var2 = np.random.random(X.shape[0]))
        X = X.assign(rd_var3 = np.random.random(X.shape[0]))
        y = self.df[self.target]

        fig, axis = plt.subplots(1, 5, figsize=(15,5))

        previous_imp_features = X.columns.tolist()

        for i,(id_train,id_test) in enumerate(CV.split(X,y)):
            model_RF.fit(X.iloc[id_train],y.iloc[id_train])

            feature_importances = model_RF.feature_importances_

            fi_df = self.create_fi_df(feature_importances,X.columns)

            top_v =fi_df[fi_df['feature_names'].str.contains('rd_').fillna(False)]['feature_importance'].idxmax()
            imp_features = fi_df.reset_index(drop=True).iloc[:top_v,:]['feature_names'].tolist()
            
            final_list_imp_features = list(set(imp_features).intersection(previous_imp_features))

            previous_imp_features = imp_features

            y_true = y.iloc[id_test]
            y_pred = model_RF.predict(X.iloc[id_test])
            acc = accuracy_score(y_true, y_pred)

            self.plot_feature_importance(fi_df, axis[i] , 'RandForest acc:{}'.format(acc))
            plt.tight_layout()

        return final_list_imp_features

    def lasso_selector(self):

        X = self.df.copy()
        X = X.drop(columns='target')
        y = self.df[self.target]

        fig, axis = plt.subplots(2, 4, figsize=(15,10))

        for i, reg_strength in enumerate(np.geomspace(0.0001, 10000, num=8)):
            model_log_reg =  LogisticRegression(penalty='l1', solver='liblinear', C=reg_strength,  random_state=self.seed)
            model_log_reg.fit(X, y)

            coefs = model_log_reg.coef_

            coefs = abs(coefs)

            coefs_sum = [sum(i) for i in zip(*coefs)]

            coef_df = self.create_fi_df(coefs_sum,X.columns)

            # return coef_df

            self.plot_feature_importance(coef_df, axis.ravel()[i] , 'log reg, C:{:.4f}'.format(reg_strength))
            plt.tight_layout()

    def kruskal_test(self):

        '''The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal.
        It is a non-parametric version of ANOVA. The test works on 2 or more independent samples, which may have 
        different sizes. Note that rejecting the null hypothesis does not indicate which of the groups differs.
        Post hoc comparisons between groups are required to determine which groups are different.'''

        categories = np.unique(self.df[self.target])

        p_values = {}

        for column in self.df.columns:
            if column != self.target:
                aux = self.df[[column, self.target]].copy()
                
                values_feat={}

                for category in categories:
                    values_feat[category] = aux[aux[self.target]== category][column].tolist()

                p_values[column] = stats.kruskal(*values_feat.values())[1]

        fig, ax = plt.subplots(1, 1, figsize=(15,10))

        sns.barplot(y=list(p_values.values()), x=list(p_values.keys()), ax=ax, orient = 'v')
        plt.title('p-values')

        for p in ax.patches:
            ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')


        return p_values


