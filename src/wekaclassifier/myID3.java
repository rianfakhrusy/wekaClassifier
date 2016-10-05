package wekaclassifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
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
        
        makeTree(toNominal(data)); //pembuatan pohon setelah data kontinu dikonversi menjadi diskrit
    }

    public void makeTree(Instances instances) throws Exception {
        // Mengecek ada tidaknya instance yang mencapai node ini
        if (instances.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = Instance.missingValue();
            m_Distribution = new double[instances.numClasses()];
            return;
        }
        // Mencari IG maksimum dari atribut
        double[] infoGains = new double[instances.numAttributes()];
        Enumeration attEnum = instances.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = computeInfoGain(instances, att);
        }
        m_Attribute = instances.attribute(Utils.maxIndex(infoGains));

        // Jika IG max = 0, buat daun dengan label kelas mayoritas
        // Jika tidak, buat successor
        if (Math.abs(infoGains[m_Attribute.index()])<1E-6)  {
            m_Attribute = null;
            m_Distribution = new double[instances.numClasses()];

            for (int i = 0; i < instances.numInstances(); i++) {
                Instance inst = (Instance) instances.instance(i);
                m_Distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
        } else {
            Instances[] splitData = splitData(instances, m_Attribute);
            m_Successors = new myID3[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                m_Successors[j] = new myID3();
                m_Successors[j].makeTree(splitData[j]);
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        System.out.println(instance);
        if (instance.hasMissingValue())
            throw new NoSupportForMissingValuesException("classifier.MyID3: This classifier can not handle missing value");
        if (m_Attribute == null)
            return m_ClassValue;
        else
            return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue())
            throw new NoSupportForMissingValuesException("classifier.MyID3: Cannot handle missing values");
        if (m_Attribute == null)
            return m_Distribution;
        else {
            if(m_Attribute.value(0).contains("<=")){
                int threshold = Integer.valueOf(m_Attribute.value(0).substring(2, 3));
                if(instance.value(m_Attribute) > threshold)
                    return m_Successors[1].distributionForInstance(instance);
                else
                    return m_Successors[0].distributionForInstance(instance);
            }
            return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
        }
    }

    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

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
    
    //mengubah instance dengan data numeric menjadi data 
    public Instances toNominal(Instances data) throws Exception {
        for (int n=0; n<data.numAttributes(); n++) {
            Attribute att = data.attribute(n);
            if (data.attribute(n).isNumeric()) {
                HashSet<Integer> uniqueValues = new HashSet();
                for (int i = 0; i < data.numInstances(); ++i)
                    uniqueValues.add((int) (data.instance(i).value(att)));
                ArrayList<Integer> dataValues = new ArrayList<>(uniqueValues);
                dataValues.sort((Integer o1, Integer o2) -> {
                    return (o1>o2)?1:-1;
                });

                // Search for threshold and get new Instances
                double[] infoGains = new double[dataValues.size() - 1];
                Instances[] tempInstances = new Instances[dataValues.size() - 1];
                for (int i = 0; i < dataValues.size() - 1; ++i) {
                    tempInstances[i] = setAttributeThreshold(data, att, dataValues.get(i));
                    infoGains[i] = computeInfoGain(tempInstances[i], tempInstances[i].attribute(att.name()));
                }
                data = new Instances(tempInstances[Utils.maxIndex(infoGains)]);
            }
        }
        return data;
    }

    private Instances setAttributeThreshold(Instances data, Attribute att, int threshold) throws Exception {
        Instances temp = new Instances(data);
        // Add thresholded attribute
        Add filter = new Add();
        filter.setAttributeName("thresholded " + att.name());
        filter.setAttributeIndex(String.valueOf(att.index() + 2));
        filter.setNominalLabels("<=" + threshold + ",>" + threshold);
        filter.setInputFormat(temp);
        Instances thresholdedData = Filter.useFilter(data, filter);

        for (int i=0; i<thresholdedData.numInstances(); i++) {
            if ((int) thresholdedData.instance(i).value(thresholdedData.attribute(att.name())) <= threshold)
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + att.name()), "<=" + threshold);
            else
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + att.name()), ">" + threshold);
        }
        Remove remove = new Remove();
        remove.setAttributeIndices(String.valueOf(att.index() + 1));
        remove.setInputFormat(thresholdedData);
        thresholdedData = Filter.useFilter(thresholdedData, remove);
        thresholdedData.renameAttribute(thresholdedData.attribute("thresholded " + att.name()), att.name());
        return thresholdedData;
    }

}
