package net.yohanes.deepsums;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DeepLearning {

    private static Logger log = LoggerFactory.getLogger(DeepLearning.class);

    private static MultiLayerNetwork model;

    private final int numRows = 4;
    private int iterations = 10;
    private int seed = 123;

    private MultiLayerNetwork getModel() {
        if (model != null) return model;
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // Locks in weight initialization for tuning
                .iterations(iterations) // # training iterations predict/classify & backprop
                .learningRate(1e-1f) // Backprop step size
                .momentum(0.1) // Speed of modifying learning rate
                .list(2) // # NN layers (doesn't count input layer)
                .layer(0, new RBM.Builder()
                        .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                        .hiddenUnit(RBM.HiddenUnit.BINARY)
                        .nIn(numRows) // # input nodes
                        .nOut(numRows) // # output nodes
                        .weightInit(WeightInit.NORMALIZED) // Weight initialization
                        .k(1) // # contrastive divergence iterations
                        .activation("sigmoid") // Activation function type
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY) // Loss function type
                        .build())
                .layer(1, new RBM.Builder()
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.GAUSSIAN)
                        .nIn(numRows) // # input nodes
                        .nOut(numRows) // # output nodes
                        .weightInit(WeightInit.NORMALIZED) // Weight initialization
                        .k(1) // # contrastive divergence iterations
                        .activation("sigmoid") // Activation function type
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY) // Loss function type
                        .build())
                .build();
        model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    public void train(String filepathTrain) {
        int numSamples = 1000;
        int batchSize = 10;
        int listenerFreq = 1;
        log.info("Load data....");
        DataSetIterator iterTrain = new DUCDataSetIterator(batchSize, numSamples, filepathTrain);
        DataSet train = iterTrain.next();
        MultiLayerNetwork trainModel = this.getModel();
        trainModel.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        trainModel.fit(train.getFeatureMatrix());
    }

    public Summary summarize(String filepathTest, String filepathSentencesTest, float[] fThreshold) throws IOException {
        int numSamples = 1000;
        int batchSize = 100;

        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        MultiLayerNetwork testModel = this.getModel();
        DataSetIterator iterTest = new DUCDataSetIterator(batchSize, numSamples, filepathTest);
        // do summarization based on threshold
        List<Integer> sentencesIds = new ArrayList<Integer>();
        int totalSentences = 0;
        int totalCorrect = 0;
        float[] maxThreshold = {0.0f, 0.0f, 0.0f, 0.0f};
        float[] averageThreshold = {0.0f, 0.0f, 0.0f, 0.0f};
        while (iterTest.hasNext()) {
            DataSet batch = iterTest.next();
            INDArray labels = batch.getLabels();
            // choose random threshold {f1, f2, f3, f4}
            INDArray o = testModel.output(batch.getFeatureMatrix());
            for (int i=0; i<o.rows();i++) {
                INDArray row = o.getRow(i);
                boolean take = true;
                for (int j=0; j<fThreshold.length; j++) {
                    // get max threshold for analysis
                    maxThreshold[j] = (maxThreshold[j] < row.getFloat(j)) ? row.getFloat(j) : maxThreshold[j];
                    averageThreshold[j] += row.getFloat(j);
                    if (row.getFloat(j) <= fThreshold[j]) {
                        // don't use sentence as summary
                        take = false;
                        break;
                    }
                }
                if (take) {
                    sentencesIds.add(i);
                    if (labels.getRow(i).getFloat(0) == 1.0f) {
                        totalCorrect++;
                    }
                }
            }
            totalSentences += o.rows();
        }

        // generate summary
        List<String> sentences = DUCUtil.getSentencesList(filepathSentencesTest);
        float[][] rawData = DUCUtil.getRawData(filepathTest);
        List<String> summary = new ArrayList<String>();
        for (int id : sentencesIds) {
            summary.add(sentences.get(id));
        }

        for (int i=0; i<averageThreshold.length; i++) {
            averageThreshold[i] = averageThreshold[i] / totalSentences;
        }

        log.info("Max threshold: " + StringUtils.join(maxThreshold, ','));
        log.info("Avg threshold: " + StringUtils.join(averageThreshold, ','));

        return new Summary(summary, rawData, sentencesIds.size(), totalCorrect);
    }

    public static void main(String[] args) throws IOException {
        DeepLearning deepLearning = new DeepLearning();
        ObjectMapper mapper = new ObjectMapper();
        Map<String,Object> conf = mapper.readValue(new File("conf.json"), Map.class);

        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        // training
        for (String filepath : (ArrayList<String>)conf.get("training")) {
            log.info("training: " + filepath);
            deepLearning.train(filepath);
        }
        //testing
        ArrayList<Map<String, Object>> results = new ArrayList<Map<String, Object>>();
        for (Map<String, Object> testing : (ArrayList<Map<String, Object>>) conf.get("testing")) {
//            log.info("testing: " + testing.get("data"));
            ArrayList<Double> recallList = new ArrayList<Double>();
            ArrayList<Double> precisionList = new ArrayList<Double>();
            ArrayList<Double> f1List = new ArrayList<Double>();
            ArrayList<Double> thresholdRaw = (ArrayList<Double>) testing.get("threshold");
            for (Double raw : thresholdRaw) {
                float[] threshold = { 0.0f, 0.0f, raw.floatValue(), 0.0f };
                Summary summary = deepLearning.summarize(
                        (String) testing.get("data"),
                        (String) testing.get("sentences"),
                        threshold);
                recallList.add(new Double(summary.getRecall()));
                precisionList.add(new Double(summary.getPrecision()));
                f1List.add(new Double(summary.getFMeasure()));
                log.info("Percentage: " + summary.getCorrectPercentage());
                log.info("TotalCorrect: " + summary.getTotalCorrect() + " / " + summary.getTotalRetrieved());
                log.info("Recall: " + summary.getRecall());
                log.info("Precision: " + summary.getPrecision());
                log.info("Fmeasure: " + summary.getFMeasure());
            }
            Map<String, Object> result = new LinkedHashMap<String, Object>();
            result.put("name", testing.get("data"));
            result.put("labels", thresholdRaw);
            result.put("recall", recallList);
            result.put("precision", precisionList);
            result.put("f1", f1List);
            results.add(result);
        }
        mapper.writeValue(new File("report.json"), results);
    }
}

class DUCDataSetIterator extends BaseDatasetIterator {
    private static final long serialVersionUID = -2022454995728680368L;
    public DUCDataSetIterator(int batch, int numExamples, String path) {
        super(batch,numExamples,new DUCDataFetcher(path));
    }

    @Override
    public boolean hasNext() {
        return fetcher.hasMore();
    }

}


class DUCDataFetcher extends BaseDataFetcher {
    private static final long serialVersionUID = 4566329799221375262L;
    public final static int NUM_EXAMPLES = 150;
    private String filepath;
    public DUCDataFetcher(String path) {
        numOutcomes = 4;
        inputColumns = 4;
        totalExamples = NUM_EXAMPLES;
        filepath = path;
        totalExamples = this.totalExamples();
    }

    @Override
    public int totalExamples() {
        int total = 0;
        try {
            total = DUCUtil.getTotalSentences(filepath);
        } catch (Exception e) {
            log.error(e.getMessage(), e);
        }
        return total;
    }

    @Override
    public boolean hasMore() {
        return cursor < totalExamples;
    }

    public void fetch(int numExamples) {
        int from = cursor;
        int to = cursor + numExamples;
        if(to > totalExamples)
            to = totalExamples;
        try {
            initializeCurrFromList(DUCUtil.loadDUC(to, from, filepath));
            cursor += numExamples;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to load duc");
        }

    }
}

class DUCUtil {

    public static List<DataSet> loadDUC(int to, int from, String filepath) throws IOException {
        FileInputStream fis = new FileInputStream(filepath);
        @SuppressWarnings("unchecked")
        List<String> lines = IOUtils.readLines(fis);
        List<DataSet> list = new ArrayList<DataSet>();
        INDArray ret = Nd4j.ones(Math.abs(to - from), 4);
        double[][] outcomes = new double[lines.size()][4];
        int putCount = 0;

        for(int i = from; i < to; i++) {
            String line = lines.get(i);
            String[] split = line.split(",");

            addRow(ret,putCount++,split);

            String outcome = split[4];
            double[] rowOutcome = new double[4];
            rowOutcome[new Float(outcome).intValue()] = 1;
            outcomes[i] = rowOutcome;
        }

        for(int i = 0; i < ret.rows(); i++) {
            int idx = (outcomes.length > (from + i)) ? from + i : outcomes.length-1;
            DataSet add = new DataSet(ret.getRow(i), Nd4j.create(outcomes[idx]));
            list.add(add);
            if (idx == (outcomes.length-1)) break;
        }
        return list;
    }

    public static int getTotalSentences(String filepath) throws IOException {
        FileInputStream fis = new FileInputStream(filepath);
        List<String> lines = IOUtils.readLines(fis);
        return lines.size();
    }

    public static void addRow(INDArray ret,int row,String[] line) {
        double[] vector = new double[4];
        for(int i = 0; i < 4; i++)
            vector[i] = Double.parseDouble(line[i]);

        ret.putRow(row,Nd4j.create(vector));
    }

    public static List<String> getSentencesList(String filepath) throws IOException {
        FileInputStream fis = new FileInputStream(filepath);
        List<String> res = IOUtils.readLines(fis);
        fis.close();
        return res;
    }

    public static float[][] getRawData(String filepath) throws IOException {
        FileInputStream fis = new FileInputStream(filepath);
        List<String> res = IOUtils.readLines(fis);
        float[][] matrix = new float[res.size()][7];
        for (int i = 0; i < matrix.length; i++) {
            String[] strArray = StringUtils.split(res.get(i), ',');
            for (int j = 0; j < strArray.length; j++) {
                matrix[i][j] = Float.parseFloat(strArray[j]);
            }
        }
        fis.close();
        return matrix;
    }

}


