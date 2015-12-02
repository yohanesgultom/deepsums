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

    public Summary(List<String> _sentences, float[][] _rawData, int _totalRetrieved, int _totalCorrect) {
        this.sentences = _sentences;
        this.rawData = _rawData;
        this.totalRetrieved = _totalRetrieved;
        this.totalCorrect = _totalCorrect;
    }

    public float getRecall() {
        return (totalRetrieved - totalCorrect) / new Float(totalRetrieved);
    }

    public float getPrecision() {
        return (totalRetrieved - totalCorrect) / new Float(totalCorrect);
    }

    public float getCorrectPercentage() {
        return totalCorrect / new Float(totalRetrieved) * 100.0f;
    }

    public float getFMeasure() {
        return (2 * this.getRecall() * this.getPrecision()) / (this.getRecall() + this.getPrecision());
    }
}
