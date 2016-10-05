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
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

public class WekaClassifier {
    
    public static void main(String[] args) {
        System.out.println("Selamat datang di Weka Classifier");
        int ch; //container angka input pilihan
        int maxch = 10; //jumlah menu pilihan
        String filedataset = ""; //nama file dataset
        Instances dataset = null; //dataset
        Classifier algo = null; //model classifier dari algoritma yang dipilih
        Scanner sc = new Scanner(System.in);
        do {
            ch = -2;
            while ( ch<-1 || ch>maxch-1 ){
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
                ch = sc.nextInt() - 1;
                sc.nextLine();
                System.out.println();
            }
            switch (ch) 
            { 
                case 0: //Load data
                    System.out.print("Masukkan nama file dataset: ");
                    filedataset = sc.nextLine();
                    //jika ekstensi file .csv ubah jadi .arff
                    if (("arff".equals(FilenameUtils.getExtension(filedataset))) ||
                            ("csv".equals(FilenameUtils.getExtension(filedataset)))){
                        if ("csv".equals(FilenameUtils.getExtension(filedataset))){
                            //load csv
                            Instances data = null;
                            CSVLoader csv = new CSVLoader();
                            try {
                                csv.setSource(new File(filedataset));
                                data = csv.getDataSet();
                            } catch (Exception e){
                                System.out.println("Error: " + e.toString());
                                System.exit(0);
                            }
                            //save file
                            ArffSaver saver = new ArffSaver();
                            String file = FilenameUtils.removeExtension(filedataset) + ".arff";
                            saver.setInstances(data);
                            try {
                                saver.setFile(new File(file));
                                saver.writeBatch();
                            } catch (IOException e) {
                                System.out.println("Error: " + e.toString());
                                System.exit(0);
                            }
                        }
                        //pilih file dataset
                        System.out.println("Dataset " + filedataset + " dipilih");
                        try {
                            dataset = new DataSource(filedataset).getDataSet(); //load from file
                            if(dataset.classIndex() == -1) //set class index if it has not been set
                                dataset.setClassIndex(dataset.numAttributes() - 1);
                        } catch (Exception e){
                            System.out.println("Error: " + e.toString());
                            System.exit(0);
                        }
                    } else
                        System.out.println("Nama file harus dengan format .csv atau .arff");
                    break; 
                case 1: //Remove attribute
                    if (dataset==null) //belum pilih atribut
                        System.out.println("Dataset belum dipilih");
                    else {
                        //select attribute
                        System.out.println("Daftar attribut: ");
                        for (int i=0;i<dataset.numAttributes();i++)
                            System.out.println((i+1) + ". " + dataset.attribute(i).name());
                        System.out.print("Pilih nomor atribut yang ingin dihapus: ");
                        int[] cc = new int[1];
                        cc[0] = sc.nextInt() - 1;
                        sc.nextLine();
                        //remove selected attribute
                        Remove remove = new Remove();
                        remove.setAttributeIndicesArray(cc);
                        String attname = dataset.attribute(cc[0]).name();
                        try {
                            remove.setInputFormat(dataset);
                            dataset = Filter.useFilter(dataset, remove);
                        } catch (Exception e){
                            System.out.println("Error: " + e.toString());
                            System.exit(0);
                        }
                        System.out.println("Atribut " + attname + " dihapus");
                    }
                    break;
                case 2: //Filter: Resample
                    if (dataset==null) //belum pilih atribut
                        System.out.println("Dataset belum dipilih");
                    else {
                        Resample resample = new Resample();
                        try {
                            resample.setInputFormat(dataset);
                            dataset = Filter.useFilter(dataset, resample);
                        } catch (Exception e) {
                            System.out.println("Error: " + e.toString());
                            System.exit(0);
                        }
                    }
                    System.out.println("Dilakukan resample pada dataset");
                case 3: //View dataset
                    System.out.println("Dataset: ");
                    for (int i=0;i<dataset.numAttributes();i++){
                        System.out.print(dataset.attribute(i).name()+"\t");
                    }
                    System.out.println();
                    for (int i=0;i<dataset.numInstances();i++){
                        for (int j=0;j<dataset.numAttributes();j++){
                            System.out.print(dataset.instance(i).toString(j)+"\t");
                        }
                        System.out.println();
                    }
                    break;
                case 4: //Build classifier
                    if (dataset==null) //belum pilih data set
                        System.out.println("Dataset belum dipilih");
                    else { //pilih algoritma
                        System.out.println("Daftar Algoritma");
                        System.out.println("1. ID3");
                        System.out.println("2. C4.5");
                        System.out.println("3. myID3");
                        System.out.println("4. myC4.5");
                        System.out.print("Pilih algoritma yang diinginkan: ");
                        int cc = sc.nextInt() - 1;
                        sc.nextLine();
                        switch (cc){
                            case 0:
                                algo = new Id3();
                                System.out.println("Algoritma ID3 dipilih");
                                break;
                            case 1:
                                algo = new J48();
                                System.out.println("Algoritma C4.5 dipilih");
                                break;
                            case 2:
                                algo = new myID3();
                                System.out.println("Algoritma myID3 dipilih");
                                break;
                            case 3:
                                algo = new myJ48();
                                System.out.println("Algoritma myC4.5 dipilih");
                                break;
                        }
                        if ((cc<0)||(cc>3)) //pilihan algoritma tidak tepat
                            System.out.println("Tidak ada algoritma yang dipilih");
                        else { //bangun classifier dari dataset
                            try {
                                algo.buildClassifier(dataset);
                                System.out.println("Classifier berhasil dibuat");
                            } catch (Exception e){
                                System.out.println("Error: " + e.toString());
                                System.exit(0);
                            }
                        }
                    }
                    break;
                case 5: //Test model for a given test set
                    //masukkan nama file input
                    System.out.print("Masukkan nama file test set: ");
                    String file = sc.nextLine();
                    System.out.println("Dataset " + file + " dipilih");
                    Instances testdata = null;
                    try {
                        testdata = new DataSource(file).getDataSet(); //load from file
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
                            labeledData.instance(i).setClassValue(algo.classifyInstance(testdata.instance(i)));
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
                            System.out.print(labeledData.instance(i).toString(j)+"\t");
                        }
                        System.out.println();
                    }
                    break; 
                case 6: //10-fold cross validation
                    if (dataset==null) //belum pilih atribut
                        System.out.println("Dataset belum dipilih");
                    else if (algo==null) //belum buat classifier
                        System.out.println("Classifier belum dibuat");
                    else {
                        Evaluation eval = null;
                        try {
                            eval = new Evaluation(dataset);
                            eval.crossValidateModel(algo, dataset, 10, new Random(1));
                        } catch (Exception e) {
                            System.out.println("Error: " + e.toString());
                            System.exit(0);;
                        }
                        System.out.println(eval.toSummaryString("==== 10-fold Cross Validation Statistics ====", false));
                    }
                    break;
                case 7: //Percentage split
                    if (dataset==null) //belum pilih atribut
                        System.out.println("Dataset belum dipilih");
                    else if (algo==null) //belum buat classifier
                        System.out.println("Classifier belum dibuat");
                    else {
                        System.out.print("Masukkan persentase : ");
                        int percent = sc.nextInt();
                        sc.nextLine();
                        System.out.println();
                        Instances splitData = new Instances(dataset);
                        splitData.randomize(new Random(1));
                        //split train and test dataset
                        int trainSize = (int) Utils.round(splitData.numInstances() * percent/100);
                        int testSize = splitData.numInstances() - trainSize;

                        Instances trainData = new Instances(splitData,0,trainSize);
                        Instances testData = new Instances(splitData,trainSize,testSize);
                        Evaluation eval = null;
                        try {
                            algo.buildClassifier(trainData);
                            eval = new Evaluation(trainData);
                            eval.evaluateModel(algo, testData);
                        } catch (Exception e){
                            System.out.println("Error: " + e.toString());
                            System.exit(0);
                        }
                        System.out.println(eval.toSummaryString("==== Test Data Statistics ====", false));
                    }
                    break;
                case 8: //Save model
                    if (dataset==null) //belum pilih atribut
                        System.out.println("Dataset belum dipilih");
                    else if (algo==null) //belum buat classifier
                        System.out.println("Classifier belum dibuat");
                    else {
                        try{
                            SerializationHelper.write(filedataset.replace(".arff", ".model"), algo);
                        } catch (Exception e){
                            System.out.println("Error: " + e.toString());
                            System.exit(0);
                        }
                        System.out.println("Model berhasil disimpan");
                    }
                    break;
                case 9: //Load model
                    System.out.print("Masukkan nama file model (.model): ");
                    try {
                        algo = (Classifier) SerializationHelper.read(sc.nextLine());
                    } catch (Exception e) {
                        System.out.println("Error: " + e.toString());
                        System.exit(0);
                    }
                    System.out.println("Model berhasil dipasang");
                    break;  
                case -1: //Exit
                    break; 
            } 
        } while (ch!=-1);
    }
}
