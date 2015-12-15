package net.yohanes.deepsums;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.StringUtils;

import java.util.*;

/**
 * Created by yohanes on 30/11/15.
 */
public class Summary {

    @Getter private List<String> sentences;
    @Getter private float[][] rawData;
    @Getter private int totalRetrieved;
    @Getter private int totalCorrect;
    @Getter private int totalCorrectExpected;

    public Summary(List<String> _sentences, float[][] _rawData, int _totalRetrieved, int _totalCorrect, int _totalCorrectExpected) {
        this.sentences = _sentences;
        this.rawData = _rawData;
        this.totalRetrieved = _totalRetrieved;
        this.totalCorrect = _totalCorrect;
        this.totalCorrectExpected = _totalCorrectExpected;
    }

    public int getTotalWrong() {
        return totalRetrieved - totalCorrect;
    }

    public float getRecall() {
        return totalCorrect / new Float(totalCorrectExpected);
    }

    public float getPrecision() {
        return totalCorrect / new Float(totalRetrieved);
    }

    public float getFMeasure() {
        return 2.0f * this.getRecall() * this.getPrecision() / (this.getRecall() + this.getPrecision());
    }

    public ArrayList<String> getSummary(String query, float compression) {
        ArrayList<String> sum = new ArrayList<String>();
        Map<Integer, Float> scores = new HashMap<Integer, Float>();
        // calculate similarity
        if (this.sentences != null) {
            // expected sentences
            int expectedSentences = Math.round(compression * rawData.length);
            for (int i = 0; i< this.sentences.size(); i++) {
                scores.put(i, this.getSimilarity(query, this.sentences.get(i)));
            }
            // sort
            Map<Integer, Float> sortedScores = new TreeMap<Integer, Float>(scores);
            int count = 0;
            for (Map.Entry<Integer, Float> entry:sortedScores.entrySet()) {
                sum.add(this.sentences.get(entry.getKey()));
                count++;
                if (count >= expectedSentences) break;
            }
        }
        return sum;
    }

    private float getSimilarity(String s1, String s2) {
        String[] s1arr = StringUtils.split(s1, ' ');
        String[] s2arr = StringUtils.split(s2, ' ');
        float wc = new Float(s1arr.length);
        int similar = 0;
        for (String ss1:s1arr) {
            for (String ss2:s2arr) {
                if (ss1.toLowerCase() == ss2.toLowerCase()) {
                    similar++;
                }
            }
        }
        return (similar > 0) ? similar / wc : 0;
    }
}
