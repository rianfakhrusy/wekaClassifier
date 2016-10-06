/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaclassifier;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FilenameUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Jasman Pardede
 */
public class TugasMLWekaClassifier {
    private Classifier algo = null; //model classifier dari algoritma yang dipilih
    
    public static void main(String[] args) {
        int ch; //container angka input pilihan
        int maxch = 10; //jumlah menu pilihan
        int minch = 1;  // jumlah menu minimum
        String filedataset = ""; //nama file dataset
        Instances dataset = null; //dataset
        TugasMLWekaClassifier mlc = new TugasMLWekaClassifier();
        Scanner sc = new Scanner(System.in);
        System.out.println("Selamat datang di Weka Classifier");
        mlc.displayGroup();
        do {
                System.out.println();
                System.out.println("Pilihan");
                System.out.println("1. Load data");
                System.out.println("2. Remove attribut");
                System.out.println("3. Filter: Resample");
                System.out.println("4. View dataset");
                System.out.println("5. Build classifier");
                System.out.println("6. Test model for a given test set");
                System.out.println("7. 10-fold cross validation");
                System.out.println("8. Percentage split");
                System.out.println("9. Save model");
                System.out.println("10. Load model");
                System.out.println("0. Exit");
                System.out.print("Masukkan pilihan : ");
                ch = sc.nextInt();
                switch(ch){
                    case 1 : {
                                System.out.print("Masukkan nama file dataset: ");
                                sc = new Scanner(System.in);
                                filedataset = sc.nextLine();
                                dataset = mlc.loadData(filedataset);
                                break;
                            }
                    case 2 : {
                                if(dataset != null)
                                    dataset = mlc.removeAttribut(dataset);
                                else{
                                    System.out.println("Dataset belum dipilih");
                                }
                                break;
                            }
                    case 3 : {
                                if(dataset != null)
                                    dataset = mlc.filterData(dataset);
                                else
                                    System.out.println("Dataset belum dipilih");
                                break;
                            }
                    case 4 : {
                                if(dataset != null)
                                    mlc.viewDataSet(dataset);
                                else
                                    System.out.println("Dataset belum dipilih");
                                break;
                            }
                    case 5 : {
                                if(dataset != null)
                                    mlc.buildClassfier(dataset);
                                else
                                    System.out.println("Dataset belum dipilih");
                                break;
                            }
                    case 6 : {
                                System.out.print("Masukkan nama file dataset: ");
                                sc = new Scanner(System.in);
                                String filetest = sc.nextLine();    
                                mlc.testModel(filetest);                                
                                break;
                            }
                    case 7 : {
                                if(dataset != null){
                                    mlc.teenFoldCrossValidate(dataset);
                                }
                                else
                                    System.out.println("Dataset belum dipilih");
                                break;
                            }
                    case 8 : {
                                if(dataset != null)
                                    mlc.percentageSplit(dataset);
                                else
                                   System.out.println("Dataset belum dipilih"); 
                                break;
                            }
                    case 9 : {
                                if(dataset != null){
                                    mlc.saveModel(filedataset);
                                }
                                else
                                    System.out.println("Dataset belum dipilih");
                                break;
                            }
                    case 10 : {
                                System.out.print("Masukkan nama file model (.model): ");
                                sc = new Scanner(System.in);
                                String fileModel = sc.nextLine();
                                mlc.loadModel(fileModel);
                                break;
                            }                    
                }
            } while(ch>=minch && ch<=maxch);
    }
    
    private Instances loadData(String filedataset){
        String extFile = "";    // ekstensi file
        Instances data = null;
        Instances dataset = null; //dataset
        
        extFile = FilenameUtils.getExtension(filedataset);
        switch(extFile){
            case "csv"  :{
                            CSVLoader csv = new CSVLoader();
                            try {
                                System.out.println("Dataset " + filedataset + " dipilih");
                                String file = FilenameUtils.removeExtension(filedataset) + ".arff";
                                csv.setSource(new File(filedataset));                               
                                data = csv.getDataSet(); 
                                // simpan ke file ext arff
                                ArffSaver saver = new ArffSaver();
                                saver.setInstances(data);
                                saver.setFile(new File(file));
                                saver.writeBatch();
                                try {
                                    dataset = new DataSource(file).getDataSet();
                                    if(dataset.classIndex() == -1)
                                        dataset.setClassIndex(dataset.numAttributes() - 1);
                                } catch (Exception ex) {
                                    Logger.getLogger(TugasMLWekaClassifier.class.getName()).log(Level.SEVERE, null, ex);
                                }
                            } catch (IOException ex) {
                                System.out.println("Dataset " + filedataset + " dipilih tidak ada.");
                            }                            
                            break;
                        }
            case "arff" :{
                            System.out.println("Dataset " + filedataset + " dipilih");
                            try {
                                dataset = new DataSource(filedataset).getDataSet();
                                if(dataset.classIndex() == -1)
                                    dataset.setClassIndex(dataset.numAttributes() - 1);
                            } catch (Exception ex) {
                                System.out.println("Dataset " + filedataset + " dipilih tidak ada.");
                            }
                            break;
                        }
            default     :{
                            System.out.println("Nama file harus dengan format .csv atau .arff");
                            break;
                        }
        }
        return dataset;
    }
   
    private Instances removeAttribut(Instances dataset){
        Scanner sc = new Scanner(System.in);
        int[] numAt;
        String input;
        boolean IsError = true;
        System.out.println("Proses remove attribut :");
        //select attribute
        if(dataset.numAttributes()>0){
            System.out.println("Daftar attribut: ");
            for(int i=0; i<dataset.numAttributes(); i++){
                System.out.println((i+1) + ". " + dataset.attribute(i).name());
            }
            do{
                System.out.print("Pilih nomor atribut yang ingin dihapus (max = "+(dataset.numAttributes()-1)+"): ");
                input = sc.nextLine();
                String ct = ""+dataset.numAttributes();
                IsError = !input.contains(ct);
            }while(!IsError);
            String[] nilaiIn = input.split("\\s+");
            numAt = new int[nilaiIn.length];
            // isi nilai
            for(int i=0; i<nilaiIn.length; i++){
                int idxR = Integer.parseInt(nilaiIn[i])-1; // indeks dimulai dari 0
                if(idxR < dataset.numAttributes())
                    numAt[i] =  idxR;
                else
                    System.out.println("Parameter Index-"+(idxR+1)+ " tidak ada!");
            }
            //remove selected attribute
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(numAt);
            
            try {
                for(int i=0; i<numAt.length; i++){
                    String attname = dataset.attribute(numAt[i]).name();
                    System.out.println("Atribut " + attname + " dihapus");
                }
                    //remove dataset
                    remove.setInputFormat(dataset);
                    dataset = Filter.useFilter(dataset, remove);         
            } catch (Exception ex) {
                Logger.getLogger(TugasMLWekaClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else{
            System.out.println("Tidak ada attribut yang dapat dihapus (di-remove)");
        }
        return dataset;
    }
    private Instances filterData(Instances dataset){
        System.out.println("Proses filter data : ");
        Resample resample = new Resample();
        try {
            resample.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, resample);
            System.out.println("Dilakukan resample pada dataset");
        } catch (Exception ex) {
            Logger.getLogger(TugasMLWekaClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        return dataset;
    }
    private void viewDataSet(Instances dataset){
        System.out.println("Proses view data set : ");
        System.out.println("Dataset : ");
        for (int i=0;i<dataset.numAttributes();i++){
            System.out.print(dataset.attribute(i).name()+"\t");
        }
        System.out.println();
        for (int i=0;i<dataset.numInstances();i++){
            for (int j=0;j<dataset.numAttributes();j++){
                System.out.print(dataset.instance(i).toString(j)+"\t" + "\t");
            }
            System.out.println();
        }
    }
    
    private void buildClassfier(Instances dataset){
        Scanner sc = new Scanner(System.in);
        Classifier myAlgo = null;
        int select;
        System.out.println("Proses build classifier :");
        System.out.println("=============== Daftar Algoritma =================");
        System.out.println("1. ID3");
        System.out.println("2. C45");
        System.out.println("3. myID3");
        System.out.println("4. myJ48");
        System.out.print("Pilih algoritma yang diinginkan : ");
        select = sc.nextInt();
        switch(select){
            case 1 : {
                        System.out.println("Algoritma ID3 dipilih");
                        myAlgo = new Id3();
                        break;
                    }
            case 2 : {
                        System.out.println("Algoritma C45 dipilih");
                        myAlgo = new J48();
                        break;
                    }
            case 3 : {
                        System.out.println("Algoritma myID3 dipilih");
                        myAlgo = new myID3();
                        break;
                    }
            case 4 : {
                        System.out.println("Algoritma myC45 dipilih");
                        myAlgo = new myJ48();
                        break;
                    }
            default :{
                        System.out.println("Tidak ada algoritma yang dipilih ([1-4])");
                        break;
                    }
        }
        if(select>=1 && select<=4){
            try {
                myAlgo.buildClassifier(dataset);
                System.out.println("Classifier berhasil dibuat");
                this.setAlgo(myAlgo);
            } catch (Exception ex) {
                Logger.getLogger(TugasMLWekaClassifier.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    private void testModel(String filedataset){
        Instances testdata = null;
        try {
            testdata = new DataSource(filedataset).getDataSet(); //load from file
            if(testdata.classIndex() == -1) //set class index if it has not been set
                testdata.setClassIndex(testdata.numAttributes() - 1);
        } catch (Exception e){
            System.out.println("Error: " + e.toString());
            System.exit(0);
        }
        //definisikan data berlabel yang menjadi output
        Instances labeledData = new Instances(testdata);
        for (int i = 0; i < labeledData.numInstances(); i++) {
            try {
                if(this.getAlgo() != null)
                    labeledData.instance(i).setClassValue(this.getAlgo().classifyInstance(testdata.instance(i)));
                else
                    this.infoAlgoNotCreate();
            } catch (Exception e) {
                System.out.println("Error: " + e.toString());
                System.exit(0);
            }
        }
        System.out.println();
        //Tampilkan data berlabel
        System.out.println("Hasil test data berlabel: ");
        for (int i=0;i<labeledData.numAttributes();i++){
            System.out.print(labeledData.attribute(i).name()+"\t");
        }
        System.out.println();
        for (int i=0;i<labeledData.numInstances();i++){
            for (int j=0;j<labeledData.numAttributes();j++){
                System.out.print(labeledData.instance(i).toString(j)+"\t\t");
            }
            System.out.println();
        }
    }
    
    private void teenFoldCrossValidate(Instances dataset){
        Evaluation eval = null;
        System.out.println("Proses 10-fold cross validate :");
        
        try {
            eval = new Evaluation(dataset);
            if(this.getAlgo() != null)
                eval.crossValidateModel(this.getAlgo(), dataset, 10, new Random(1));
            else
                this.infoAlgoNotCreate();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error: " + e.toString());
            System.exit(0);;
        }
        System.out.println(eval.toSummaryString("==== 10-fold Cross Validation Statistics ====", false));
                    
    }
    
    private void percentageSplit(Instances dataset){
        Scanner sc = new Scanner(System.in);
        System.out.print("Masukkan persentase : ");
        int percent = sc.nextInt();
        Instances splitData = new Instances(dataset);
        splitData.randomize(new Random(1));
        //split train and test dataset
        int trainSize = (int) Utils.round(splitData.numInstances() * percent/100);
        int testSize = splitData.numInstances() - trainSize;

        Instances trainData = new Instances(splitData,0,trainSize);
        Instances testData = new Instances(splitData,trainSize,testSize);
        Evaluation eval = null;
        try {
            if(this.getAlgo() != null){
                algo.buildClassifier(trainData);
                eval = new Evaluation(trainData);
                eval.evaluateModel(algo, testData);
            }
            else
                this.infoAlgoNotCreate();
        } catch (Exception e){
            System.out.println("Error: " + e.toString());
            System.exit(0);
        }
        System.out.println(eval.toSummaryString("==== Test Data Statistics ====", false));  
    }
    
    private void saveModel(String filedataset){
        System.out.println("Proses Save model :");
        try {
            if(this.getAlgo() != null){
                SerializationHelper.write(filedataset.replace(".arff", ".model"), this.getAlgo());
                System.out.println("Model berhasil disimpan");
            }
            else
                infoAlgoNotCreate();
        } catch (Exception ex) {
            //Logger.getLogger(TugasMLWekaClassifier.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Model gagal disimpan");
        }
    }

    private void infoAlgoNotCreate(){
        System.out.println("Classifier belum dipilih");
        System.out.println("Tentukan algoritma yang diinginkan (Pilih step [5])");
    }
    
    private void loadModel(String fileModel){
        Classifier myAlgo = null;
        System.out.println("Proses load model :");
        try {
            myAlgo = (Classifier) SerializationHelper.read(fileModel);
            if(myAlgo != null){
                this.setAlgo(myAlgo);
                System.out.println("Model berhasil dipasang");
            }
            else
                System.out.println("Model gagal disimpan");
        } catch (Exception e) {
            System.out.println("Error: " + e.toString());
            System.exit(0);
        }       
    }
    
    private void displayGroup(){
        System.out.println("====================================================");
        System.out.println("================TUGAS CLASSIFIER ===================");
        System.out.println("============ 33216013 : Jasman Pardede =============");
        System.out.println("============ 13511001 : Thea Olivia ================");
        System.out.println("============ 13511008 : Muhammad Rian Fakhrusy =====");
        System.out.println("====================================================");    
    }

    public Classifier getAlgo() {
        return algo;
    }

    public void setAlgo(Classifier algo) {
        this.algo = algo;
    }        
}   
