package net.yohanes.deepsums;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

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
        return (2 * this.getRecall() * this.getPrecision()) / (this.getRecall() + this.getPrecision());
    }
}
