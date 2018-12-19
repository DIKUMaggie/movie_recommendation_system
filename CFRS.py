import numpy as np
import scipy.spatial.distance as dist
import math
import sys
import os
import time
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from scipy.stats import f_oneway

# users = 671
# items = 9125

class cfrs():
    def __init__(self,movie,users=671,items=9125):
        """
        init map movies to ids
        :param movie: movie file
        :param users: number of users
        :param items: number of movies

        """

        self.users = users
        self.items = items
        self.genresDict = TfidfVectorizer()
        self.movieGenresInit(movie)


    def movieGenresInit(self,movie):
        """
        movie map id and tf-idf statistic of movie attribute corpus
        :param movie: the  attribute corpus of movies
        :return: None
        """
        print("init")
        corpus = []
        self.movieMapId = dict()
        self.genres = []
        if not os.path.exists(movie):
            print("input file does not exist")
            sys.exit(-1)
        genres = []
        with open(movie, "r") as fr:
            lines = fr.readlines()
            lines.pop(0)
            for i, line in enumerate(lines):
                line = line.strip()
                line = line.split(",")
                self.movieMapId[int(line[0])] = i + 1
                if line[-1] == "(no genres listed)":
                    corpus.append("XX")
                else:
                    corpus.append(" ".join(line[-1].split("|")))

        self.genresDict.fit_transform(corpus)
        for line in lines:
            line = line.strip()
            line = line.split(",")
            if line[-1] == "(no genres listed)":
                genres.append(self.genresDict.transform(["XX"]).toarray()[0])
            else:
                genres.append(self.genresDict.transform([" ".join(line[-1].split("|"))]).toarray()[0])
        self.genres = np.array(genres)


    def getDataFromFileCSV(self,input):
        """
        convert the dataset to vectors for ml-latest-small
        :param input: file of ratings.csv
        :return: dataset vectors
        """
        data = []
        if not os.path.exists(input):
            print("input file does not exist")
            sys.exit(-1)
        with open(input, "r") as fr:
            lines = fr.readlines()
            lines.pop(0)
            for line in lines:
                line = line.strip()
                line = line.split(",")
                line = [int(line[0]), int(line[1]), float(line[2])]
                data.append(line)
        return data

    def splitFromEachUser(self,k,num_line):
        """
        get k part ratings from one user
        :param k: k part
        :param num_line: the ratings belongs to one user
        :return: k part ratings one user
        """
        interval = int(len(num_line)/k)
        ret = []
        for i in range(k):
            if i==k-1:
                ret.append(num_line[i*interval:])
            else:
                ret.append(num_line[i*interval:(i+1)*interval])
        return ret


    def splitTrainAndTest(self, num_line, k=10):
        """
        get k fold dataset
        :param num_line: dataset
        :param k: fold
        :return: k fold dataset [da1,da2,da3,da4...dak];da1=[train,test]
        """
        start = 0
        end = 0
        users = []
        for i, line in enumerate(num_line):
            if i == len(num_line) - 1:
                users.append(self.splitFromEachUser(k, newSet[start:]))
            elif num_line[end][0] == line[0]:
                end = i
            else:
                users.append(self.splitFromEachUser(k, newSet[start:i]))
                start = i
                end = i
        dataset =[]
        for i in range(k):
            train = []
            test = []
            for user in users:
                for j in range(k):
                    if i!=j:
                        train += user[j]
                test += user[i]
            dataset.append((train,test))
        return dataset

    def adjustCosine(self,data1,data2):
        """
        adjusted cosine similarity
        :param data: (users,items)
        :return: similarity (items,items)
        """
        ind1 = (data1 == 0)
        data2[ind1] = 0
        ind2 = (data2 != 0)
        if len(data1[ind2])==0:
            return 0
        return 1-dist.cosine(data1[ind2],data2[ind2])

    def calItemToAll(self,item):
        sim = np.zeros((1,self.items))[0]
        if not np.count_nonzero(self.data[:, item]):
            return sim
        for i in range(self.items):
             if np.count_nonzero(self.data[:,i]):
               sim[i] = self.adjustCosine(self.data[:,item],self.data[:,i])
        return sim

    def predictRateOfUser(self,user,movie):
        """
        predict the rating
        :param user: the audience
        :param movie: movie to be rated
        :return: the prediction of rating
        """
        #initialize the rating to be the average value of 5-stars 
        predict = 2.75
        if np.count_nonzero(self.Mat[:, movie]):
            if movie in self.sim:
                sim = self.sim[movie]
            else:
                sim = self.calItemToAll(movie)
                self.sim[movie] = sim
            ind = (self.Mat[user] > 0)
            normal = np.sum(np.absolute(sim[ind]))
            if normal > 0:
                predict = np.dot(self.Mat[user],sim) / normal
                if predict < 0.5:
                    predict = 0.5
                if predict > 5:
                    predict = 5
        return predict

    def testItemSimilar(self,data,itemi,itemj):
        similarity = np.zeros((self.items,self.items))
        print(self.adjustCosine(data[:, itemi], data[:, itemj]))
        print(similarity[itemi][itemj])
        
    def testSim(self,data,itemi,itemj):

        self.Mat = np.zeros((self.users, self.items))
        for line in data:
            self.Mat[line[0] - 1][self.movieMapId[line[1]] - 1] = line[2]
        print(self.Mat)
        self.Mat[self.Mat == 0] = np.nan
        self.M_mean = np.nanmean(self.Mat, axis=1)
        self.data = np.nan_to_num(self.Mat - self.M_mean[:, None])
        here = time.time()
        self.testItemSimilar(self.data,itemi,itemj)
        print(time.time()-here)


    def initUserAndRating(self,data):
        """
        init users*movies matrix of ratings
        :param data: init (users,movier)
        :return: matrix
        """
        self.Mat = np.zeros((self.users, self.items))
        for line in data:
            self.Mat[line[0] - 1][self.movieMapId[line[1]] - 1] = line[2]
        return self.Mat

    def recommandNewItem(self,data,item):
        """
        recommand the new item to the first 100 users
        :param data: dataset of ratings
        :param item: movie attribute
        :return: None
        """
        item = item.split(",")[-1]
        itemToVect = self.genresDict.transform([" ".join(item.split("|"))]).toarray()[0]
        genresSimilairty = []
        for distance in self.genres:
            genresSimilairty.append(1-dist.cosine(distance,itemToVect))
        mat = self.initUserAndRating(data)
        rating = np.dot(mat,np.array(genresSimilairty))/np.sum(genresSimilairty)
        print(sorted(np.argsort(-rating)[:100]))

    def cross_validation(self, data, k=10):
        """
         cross validation
        :param data: dataset of ratings
        :param k: k fold
        :return: predictions of dataset
        """
        newData = shuffle(data,random_state =200)
        dataset = self.splitTrainAndTest(newData, k)
        predict =[]
        cout = 0
        start = time.time()
        MAE = []
        RMSE = []
        processTime = 0
        print("Start to process %s fold cross validation"%k)
        for train,test in dataset:
            self.sim = dict()
            print("Cross:",cout)
            print(len(test))
            cout+=1
            self.Mat = np.zeros((self.users, self.items))
            for line in data:
                self.Mat[line[0] - 1][self.movieMapId[line[1]] - 1] = line[2]
            self.Mat[self.Mat == 0] = np.nan
            self.M_mean = np.nanmean(self.Mat, axis=1)
            self.data = np.nan_to_num(self.Mat - self.M_mean[:, None])
            self.Mat = np.nan_to_num(self.Mat)
            start = time.time()
            
            testPred = []
            for i, item in enumerate(test):
                predict.append(self.predictRateOfUser(item[0] - 1, self.movieMapId[item[1]] - 1))
                testPred.append(self.predictRateOfUser(item[0] - 1, self.movieMapId[item[1]] - 1))    
                #if i%2000==0:
                    #print(item[0], item[1], predict[-1], item[2])
            processTime += time.time() - start
            MAE.append(self.evaluateByMAE(testPred,test))
            RMSE.append(self.evaluateByRMSE(testPred,test))
        print("Processing time: ",processTime)
        return predict,MAE,RMSE

    def evaluateByMAE(self,predict,test):
        """
        evaluate by MAE
        :param predict: prediction of all the instances(cross validation)
        :param test: test here is all the instances(cross validation)
        :return:
        """
        MAE = 0.0
        for i,item in enumerate(test):
            MAE += abs(predict[i]-item[2])
        return MAE/len(test)

    def evaluateByRMSE(self,predict,test):
        """
        evaluate by RMSE
        :param predict: prediction of all the instances(cross validation)
        :param test: test here is all the instances(cross validation)
        :return:
        """
        RMSE = 0.0
        for i,item in enumerate(test):
            RMSE += (predict[i]-item[2])**2
        return math.sqrt(RMSE/len(test))

if __name__ =="__main__":

    solution = cfrs("ml-latest-small/movies.csv")

    # get dataset  from "ml-latest-samll/ratings.csv"
    dataset = solution.getDataFromFileCSV("ml-latest-small/ratings.csv")

    # test the sim
    #solution.initUserAndRating(dataset)

    # 3 fold cross validation to get the prediction len(prediction)==len(test)
    #predThree,maeThree,rmseThree = solution.cross_validation(dataset,3)
    #print("k=3, MAE: %0.3f, RMSE: %0.3f"%(np.mean(maeThree),np.mean(rmseThree))) 

    predTen,maeTen,rmseTen = solution.cross_validation(dataset,10)  # 10-fold   
    print("k=10, MAE: %0.3f, RMSE: %0.3f"%(np.mean(maeTen),np.mean(rmseTen)))

    #print("MAE: f_val:%s, p_val:%s"%f_oneway(maeThree,maeTen))
    #print("RMSE: f_val:%s, p_val:%s"%f_oneway(rmseThree,rmseTen))

    # new item recommend
    newItem1 = "Paper Towns (2012),Fantasy|Film-Noir"
    solution.recommandNewItem(dataset,newItem1)
    newItem2 = "Interstellar (2014),Sci-Fi|IMAX"
    solution.recommandNewItem(dataset,newItem2)
    newItem3 = "Wrinkles (Arrugas) (2011),Animation|Drama"
    solution.recommandNewItem(dataset,newItem3)

