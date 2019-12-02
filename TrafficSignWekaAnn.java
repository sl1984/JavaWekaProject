import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class TrafficSignWekaAnn {

	
	public static void main(String[] args) {
		System.out.println("Processing started...");
		TrafficSignWekaAnn wekaann = new TrafficSignWekaAnn();
		
		//file paths & source file names
		String filepath="/Users/sethulekshmy/eclipse-workspace/DataMiningProject/data/";
		String xfileName_train = "x_train_gr_smpl.csv";
		String yfileName_train = "y_train_smpl.csv";
		String outputfile_train = "merged_xy_train_gr_smpl.arff";
		String xfileName_test = "x_test_gr_smpl.csv";
		String yfileName_test = "y_test_smpl.csv";
		String outputfile_test = "merged_xy_test_gr_smpl.arff";
		
		
		 Instances trainDataSet,  testDataSet = null;
		 try {
			
			 
			
			// build Instances of original training dataset from CSV file
			trainDataSet = wekaann.buildDataSet(filepath, xfileName_train, yfileName_train, outputfile_train); //original file
			
			// build Instances of original test dataset from CSV file
			testDataSet = wekaann.buildDataSet(filepath, xfileName_test, yfileName_test, outputfile_test); //original file
			 
						
			StringBuffer result = new StringBuffer();
			
			// Apply Classifiers - Invoking the MLP using a set of options for experimenting the impact on performance
			String[] options1 = new String[] { "-L", "0.1","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options1));
			
			String[] options2 = new String[] { "-L", "0.5","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options2));
			
			String[] options3 = new String[] { "-L", "1","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options3));
			
			String[] options4 = new String[] { "-L", "0.3","-M", "0.1","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options4));
			
			String[] options5 = new String[] { "-L", "0.3","-M", "0.5","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options5));
			
			String[] options6 = new String[] { "-L", "0.3","-M", "1","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options6));
			
			String[] options7 = new String[] { "-L", "0.1","-M", "0.1","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options7));
			
			String[] options8 = new String[] { "-L", "0.1","-M", "0.1","-N", "10","-V", "0", "-S", "0","-E", "50", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options8));
			
			String[] options9 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options9));
			
			String[] options10 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "20"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options10));
			
			String[] options11 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "40"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options11));
			
			String[] options12 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "60"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options12));
			
			String[] options13 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10,10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options13));
			
			String[] options14 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "20,20"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options14));
			
			String[] options15 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "5,5,5"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options15));
			
			String[] options16 = new String[] { "-L", "0.3","-M", "0.2","-N", "10","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options16));
			
			String[] options17 = new String[] { "-L", "0.3","-M", "0.2","-N", "20","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options17));
			
			String[] options18 = new String[] { "-L", "0.3","-M", "0.2","-N", "50","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options18));
			
			// write the results to a  log file for analysis
			wekaann.writeToFile(result.toString(), filepath+"/mlpresults_orig12k.log");
			
			
			
			//Now run MLP on the Simple reduced 4000 train and test ARFF
			
			trainDataSet = wekaann.getInstancesFromArffFile(filepath+"x_train_gr_smplReduced-4000.arff");
			trainDataSet.setClassIndex(trainDataSet.numAttributes() - 1);
			testDataSet = wekaann.getInstancesFromArffFile(filepath+"x_test_gr_smplReduced-4000.arff");
			testDataSet.setClassIndex(testDataSet.numAttributes() - 1);
			
			StringBuffer result1 = new StringBuffer(); 
			
			
			
			// Apply Classifiers - Invoking the MLP using a set of options for experimenting the impact on performance
			String[] options21 = new String[] { "-L", "0.1","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options21));
			
			String[] options22 = new String[] { "-L", "0.5","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options22));
			
			String[] options23 = new String[] { "-L", "1","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options23));
			
			String[] options24 = new String[] { "-L", "0.3","-M", "0.1","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options24));
			
			String[] options25 = new String[] { "-L", "0.3","-M", "0.5","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options25));
			
			String[] options26 = new String[] { "-L", "0.3","-M", "1","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options26));
			
			String[] options27 = new String[] { "-L", "0.1","-M", "0.1","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options27));
			
			String[] options28 = new String[] { "-L", "0.1","-M", "0.1","-N", "500","-V", "0", "-S", "0","-E", "50", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options28));
			
			String[] options29 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options29));
			
			String[] options30 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "20"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options30));
			
			String[] options31 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "40"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options31));
			
			String[] options32 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "60"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options32));
			
			String[] options33 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10,10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options33));
			
			String[] options34 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "20,20"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options34));
			
			String[] options35 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "5,5,5"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options35));
			
			String[] options36 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options36));
			
			String[] options37 = new String[] { "-L", "0.3","-M", "0.2","-N", "1000","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options37));
			
			String[] options38 = new String[] { "-L", "0.3","-M", "0.2","-N", "2000","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result1.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options38));
			
			// write the results to a log file for analysis
			wekaann.writeToFile(result1.toString(), filepath+"/mlpresults_simple4k.log");
			
			
			//Finally read the Simple reduced 9000 train and test ARFF
			
			trainDataSet = wekaann.getInstancesFromArffFile(filepath+"x_train_gr_smplReduced-9000.arff");
			trainDataSet.setClassIndex(trainDataSet.numAttributes() - 1);
			testDataSet = wekaann.getInstancesFromArffFile(filepath+"x_test_gr_smplReduced-9000.arff");
			testDataSet.setClassIndex(testDataSet.numAttributes() - 1);
			
			StringBuffer result2 = new StringBuffer(); 
			
			
			// Apply Classifiers - Invoking the MLP using a set of options for experimenting the impact on performance
			String[] options121 = new String[] { "-L", "0.1","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options121));
			
			String[] options122 = new String[] { "-L", "0.5","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options122));
			
			String[] options123 = new String[] { "-L", "1","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options123));
			
			String[] options124 = new String[] { "-L", "0.3","-M", "0.1","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options124));
			
			String[] options125 = new String[] { "-L", "0.3","-M", "0.5","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options125));
			
			String[] options126 = new String[] { "-L", "0.3","-M", "1","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options126));
			
			String[] options127 = new String[] { "-L", "0.1","-M", "0.1","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options127));
			
			String[] options128 = new String[] { "-L", "0.1","-M", "0.1","-N", "500","-V", "0", "-S", "0","-E", "50", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options128));
			
			String[] options129 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options129));
			
			String[] options130 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "20"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options130));
			
			String[] options131 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "40"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options131));
			
			String[] options132 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "60"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options132));
			
			String[] options133 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "20,20"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options133));
			
			String[] options134 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "40,40"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options134));
			
			String[] options135 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "20,20,20"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options135));
			
			String[] options136 = new String[] { "-L", "0.3","-M", "0.2","-N", "500","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options136));
			
			String[] options137 = new String[] { "-L", "0.3","-M", "0.2","-N", "1000","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options137));
			
			String[] options138 = new String[] { "-L", "0.3","-M", "0.2","-N", "2000","-V", "0", "-S", "0","-E", "20", "-H", "10"}; 
			result2.append(wekaann.ApplyMLP(trainDataSet, testDataSet, options138));
			
			// write results to a log file for analysis
			wekaann.writeToFile(result2.toString(), filepath+"/mlpresults_simple9k.log");
			
			
			System.out.println("Processing completed...");
			
	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 
		 

	}
	
	
	/**
	*  This method will return the training data set
	* 
	* @param file
	*/
	public Instances buildDataSet(String filepath, String xfile, String yfile, String outputfile) throws Exception {
		
		// Step 1 - Read data into weka Instances from the CSV file.
		Instances dataSet = getInstancesFromCsvFile(filepath+xfile);
		System.out.println("Attribute count of trainDataSet= "+dataSet.numAttributes());
		
		//Step 2- Read the class label file (from Y file) and merge with above Instance using filter	
		Instances dataSet_cls_lbl = getInstancesFromCsvFile(filepath+yfile);
		
		// Step 3- merging attributes from Y file onto existing Instances created from X file.. 
		//The class label is added as last attribute to the current Instance.
	    Add filterAdd = new Add();
	    filterAdd.setAttributeIndex("last");
	    filterAdd.setAttributeName("sign");
	    filterAdd.setInputFormat(dataSet);
	    dataSet = Filter.useFilter(dataSet, filterAdd);
        
        for (int i = 0; i < dataSet.numInstances(); i++) {
        	dataSet.instance(i).setValue(dataSet.numAttributes() - 1, dataSet_cls_lbl.instance(i).value(0));
        }
        
        System.out.println("Attribute count of dataSet after merging class label from Y file= "+dataSet.numAttributes());
        
		//Step 4 - convert the Instances into a ARFF format and write to output file
		convertInstancesToArffFile(dataSet, filepath+outputfile);
		System.out.println("Merged ARFF file is saved to: "+filepath+outputfile);
		dataSet.setClassIndex(dataSet.numAttributes() - 1);
		dataSet = filterNumbericToNominal(dataSet,"last");
		return dataSet;

	}
	
	
	/**
	*  This method will read data from CSV file and returns the weka Instances
	* 
	* @param file
	*/
	public Instances getInstancesFromCsvFile(String infile) throws Exception {
		
		// build Instances from CSV files
	    CSVLoader loader = new CSVLoader(); 
	    loader.setSource(new File(infile));
	    
	    Instances instances = loader.getDataSet();
	    return instances;    

	}
	
	/**
	*  This method will read data from ARFF file and returns the weka Instances
	* 
	* @param file
	*/
	public Instances getInstancesFromArffFile(String infile) throws Exception {
		
		// build Instances from CSV files
	    ArffLoader loader = new ArffLoader(); 
	    loader.setSource(new File(infile));
	    
	    Instances instances = loader.getDataSet();
	    return instances;    

	}
	 
	 /**
	 *  This method will write Instances data to ARFF file
	 * 
	 * @param file
	 */
	 public void convertInstancesToArffFile(Instances instances, String outfile) throws Exception {
		
		// convert Instances to ARFF file
		ArffSaver arffsaver = new ArffSaver();
		arffsaver.setInstances(instances);
		arffsaver.setFile(new File(outfile));
		arffsaver.writeBatch();    

	 }
	 
	 /**
	 *  This method will apply numeric to nominal filter
	 * 
	 * @param instances
	 * @param attributeIndex
	 */
	 public Instances filterNumbericToNominal(Instances instances, String attributeIndex) throws Exception {
		
		//Applying Filter - Numeric to Nominal
        System.out.println("Applying Filter - Numeric to Nominal filter on attribute index "+attributeIndex);
        
        Instances nominalData = null;
        nominalData = new Instances(instances);
        
        NumericToNominal filterNominal= new NumericToNominal();
        filterNominal.setOptions(new String[] { "-R", attributeIndex});
        filterNominal.setInputFormat(nominalData);
        nominalData=Filter.useFilter(nominalData, filterNominal);  
        return nominalData;

	 }
	 
	 /**
	 *  This method will apply filter - Best Attribute Selector
	 * 
	 * @param instances
	 */
	 public Instances FilterBestAttributeSelector(Instances instances) throws Exception {
		
		//Applying Filter - Best Attribute selection
        System.out.println("Applying Filter - Best Attribute selection");
        
        Instances nominalData = null;
        nominalData = new Instances(instances);
        
        AttributeSelection filterAttributeSelection = new AttributeSelection();
	    
	    CfsSubsetEval eval = new CfsSubsetEval();
	    eval.setOptions(new String[] { "-P", "1" , "-E", "1"});
        
	    BestFirst search = new BestFirst();
	    search.setOptions(new String[] { "-D", "1" , "-N", "5"});
	    
	    
	    filterAttributeSelection.setEvaluator(eval);
	    filterAttributeSelection.setSearch(search);
	    
	    filterAttributeSelection.setInputFormat(nominalData);
	    nominalData=Filter.useFilter(nominalData, filterAttributeSelection);
	    
	    System.out.println("Best attribute count="+nominalData.numAttributes());
	    
	    return nominalData;

	 }
	 
	 /**
	 *  This method will apply MLP Classifier
	 * 
	 * @param instances
	 */
	 public String ApplyMLP(Instances trainData, Instances testData, String[] classifierOption) throws Exception {
        
		String options = Arrays.toString(classifierOption);
		System.out.println("Running MLP with options\n"+options);
		 
		MultilayerPerceptron mlpClassifier = new MultilayerPerceptron();
		mlpClassifier.setOptions(classifierOption);
		mlpClassifier.buildClassifier(trainData);
		
		
		
		Evaluation evaluation = new Evaluation(trainData);
		evaluation.evaluateModel(mlpClassifier, testData);
		 
	    StringBuffer result;

	    result = new StringBuffer();
	    result.append("******* MLP Results - Options:"+options+"\n");
	     try {
	    	 result.append(evaluation.toSummaryString() + "\n");
	    	 result.append("Avg Cost="+evaluation.avgCost() + "\n");
	    	 result.append("Total Cost="+evaluation.totalCost() + "\n");
	    	 result.append(evaluation.toMatrixString("Confusion Matrix") + "\n");
	         result.append(evaluation.toClassDetailsString("Class Details String") + "\n");
	        
	      } catch (Exception e) {
	        e.printStackTrace();
	      }

	      System.out.println("Classifier Outputs\n"+result.toString());
	      return result.toString();
	    
	  }
	 
	 /**
	  *  This method will write results to a file
	  * 
	  * @param instances
	  */
	 public void writeToFile(String content, String fileName) throws IOException {
		 BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		 writer.write(content);
		 writer.close();
	}

}
