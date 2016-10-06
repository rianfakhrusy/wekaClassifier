/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaclassifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import weka.core.Utils;

public class myID3 extends Classifier {

    private myID3[] m_Successors; //simpul anak
    private Attribute m_Attribute; //atribut pada akar pohon
    private double m_ClassValue; //class value pada daun
    private double[] m_Distribution; //class distribution pada daun

    //hal yang bisa ditangani classifier
    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();
        //atribut
        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        //class
        capabilities.enable(Capability.BINARY_CLASS);
        capabilities.enable(Capability.NOMINAL_CLASS);
        capabilities.enable(Capability.MISSING_CLASS_VALUES);
        capabilities.enable(Capability.MISSING_VALUES);
        //instances
        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }
    
    //membangun classifier
    @Override
    public void buildClassifier(Instances data) throws Exception {
        //pengujian apakah data dapat ditangani
        getCapabilities().testWithFail(data);
        
        //menghapus instance dengan missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        makeTree(NumerictoNominal(data)); //pembuatan pohon setelah data kontinu dikonversi menjadi diskrit
    }
    
    //membuat pohon keputusan
    public void makeTree(Instances data) throws Exception {
        if (data.numInstances() == 0) { //jika instance kosong
            m_Attribute = null;
            m_ClassValue = Instance.missingValue();
            m_Distribution = new double[data.numClasses()];
            return;
        }
        // Mencari atribut dengan information gain maskimal
        double[] infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = computeInfoGain(data, att);
        }
        m_Attribute = data.attribute(Utils.maxIndex(infoGains));

        if (Math.abs(infoGains[m_Attribute.index()])<1E-6)  { //Information gain mendekati nol, node adalah daun
            m_Attribute = null;
            m_Distribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                m_Distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
        } else { //pembuatan node anak dari pohon
            Instances[] splitData = splitData(data, m_Attribute);
            m_Successors = new myID3[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                m_Successors[j] = new myID3();
                m_Successors[j].makeTree(splitData[j]);
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        if (m_Attribute == null)
            return m_ClassValue;
        else
            return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        if (m_Attribute == null)
            return m_Distribution;
        else {
            if(m_Attribute.value(0).contains("<")){
                int threshold = Integer.valueOf(m_Attribute.value(0).substring(1, 2));
                if(instance.value(m_Attribute) > threshold)
                    return m_Successors[1].distributionForInstance(instance);
                else
                    return m_Successors[0].distributionForInstance(instance);
            }
            return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
        }
    }
    
    //menghitung entropy
    private double computeEntropy(Instances data) throws Exception {
        double[] classCounts = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); ++i)
            classCounts[(int) data.instance(i).classValue()]++;
        double entropy = 0;
        for (int i = 0; i < classCounts.length; i++) 
            if (classCounts[i] > 0)
                entropy -= classCounts[i]/data.numInstances() * Math.log(classCounts[i]/data.numInstances()) / Math.log(2);
        return entropy;
    }
    
    //menghitung information gain
    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0)
                infoGain -= ((double) splitData[j].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[j]);
        }
        return infoGain;
    }
    
    //pemisahan data menjadi sejumlah banyaknya nilai dari atribut nominal input
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++)
            splitData[i] = new Instances(data, data.numInstances());
        for (int i = 0; i < data.numInstances(); i++)
            splitData[(int) data.instance(i).value(att)].add(data.instance(i));
        for (int i = 0; i < splitData.length; i++)
            splitData[i].compactify();
        return splitData;
    }
    
    //mengubah instance dengan data numeric menjadi data nominal
    public Instances NumerictoNominal(Instances data) throws Exception {
        for (int n=0; n<data.numAttributes(); n++) {
            Attribute att = data.attribute(n);
            if (data.attribute(n).isNumeric()) { //ubah atribut jika kontinu
                HashSet<Integer> uniqueValues = new HashSet();
                for (int i = 0; i < data.numInstances(); ++i)
                    uniqueValues.add((int) (data.instance(i).value(att)));
                //urutkan nilai yang unik dari atribut
                ArrayList<Integer> dataValues = new ArrayList<>(uniqueValues);
                dataValues.sort((Integer o1, Integer o2) -> {
                    return (o1>o2)?1:-1;
                });
                //cari pemisah antar nilai yang memaksimalkan information gain
                double[] infoGains = new double[dataValues.size() - 1]; //infogain pada setiap split
                Instances[] tempInstances = new Instances[dataValues.size() - 1];
                for (int i = 0; i < dataValues.size() - 1; ++i) { //buat atribut baru dan hapus atribut lama
                    Instances temp = new Instances(data);
                    //buat atribut baru berdasarkan nilai threshold
                    Add filter = new Add();
                    filter.setAttributeName("thresholded " + att.name());
                    filter.setAttributeIndex(String.valueOf(att.index() + 2));
                    filter.setNominalLabels("<" + dataValues.get(i) + ",>=" + dataValues.get(i));
                    filter.setInputFormat(temp);
                    tempInstances[i]  = Filter.useFilter(data, filter);
                    //pisahkan atribut menjadi dua bagian sesuai dengan threshold
                    for (int j=0; j<tempInstances[i].numInstances(); j++) 
                        tempInstances[i].instance(j).setValue(tempInstances[i].attribute("thresholded " + att.name()), ((int) tempInstances[i].instance(j).value(tempInstances[i].attribute(att.name()))<dataValues.get(i)?"<":">=") + dataValues.get(i));
                    //hapus atribut yang lama
                    Remove remove = new Remove();
                    remove.setAttributeIndices(String.valueOf(att.index() + 1));
                    remove.setInputFormat(tempInstances[i] );
                    tempInstances[i]  = Filter.useFilter(tempInstances[i] , remove);
                    tempInstances[i].renameAttribute(tempInstances[i].attribute("thresholded " + att.name()), att.name());
                    infoGains[i] = computeInfoGain(tempInstances[i], tempInstances[i].attribute(att.name()));
                }
                data = new Instances(tempInstances[Utils.maxIndex(infoGains)]);
            }
        }
        return data;
    }
}
