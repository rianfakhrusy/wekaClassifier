/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaclassifier;

import java.util.Scanner;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author ASUS
 */
public class WekaClassifier {

    /**
     * @param args the command line arguments
     */
    
    public static void main(String[] args) {
        System.out.println("Selamat datang di Weka Classifier");
        int ch;
        int maxch = 9;
        String filedataset = "";
        Instances dataset = null;
        Scanner sc = new Scanner(System.in);
        do {
            ch = -1;
            while ( ch<0 || ch>maxch-1 ){
                System.out.println();
                System.out.println("Pilihan");
                System.out.println("1. Load data");
                System.out.println("2. Remove attribut");
                System.out.println("3. Filter: Resample");
                System.out.println("4. Build classifier");
                System.out.println("5. Test model for a given test set");
                System.out.println("6. Save model");
                System.out.println("7. Load model");
                System.out.println("8. Classify unseen data from a given model");
                System.out.println("9. Exit");
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
                    System.out.println("Dataset " + filedataset + " dipilih");
                    try {
                        dataset = new DataSource(filedataset).getDataSet(); //load from file
                        if(dataset.classIndex() == -1) //set class index if it has not been set
                            dataset.setClassIndex(dataset.numAttributes() - 1);
                    } catch (Exception e){
                        e.printStackTrace();
                    }
                    break; 
                case 1: //Remove attribute
                    if (dataset==null)
                        System.out.println("Dataset belum dipilih");
                    else {
                        for (int i=0;i<dataset.numAttributes();i++)
                            System.out.println(dataset.attribute(i).name());
                    }
                    break;
                case 2: //Filter: Resample
                    System.out.println(3);
                    break; 
                case 3: //Build classifier
                    System.out.println(4);
                    break;
                case 4: //Test model for a given test set
                    System.out.println(5);
                    break; 
                case 5: //Save model
                    System.out.println(6);
                    break;
                case 6: //Load model
                    System.out.println(7);
                    break; 
                case 7: //Classify unseen data from a given model
                    System.out.println(8);
                    break;
                case 8: //Exit
                    System.out.println(9);
                    break; 
            } 
        } while (ch!=maxch-1);
    }
}
