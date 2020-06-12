package edu.ntu.pp2011;

import java.io.IOException;
import java.util.StringTokenizer;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class WordCount {
  /* mapper */
  public static class Map extends MapReduceBase 
    implements Mapper<LongWritable, Text, Text, IntWritable> {
    public void map(LongWritable key, Text value, 
		    OutputCollector<Text, IntWritable> output, 
		    Reporter reporter) 
      throws IOException {
      StringTokenizer stk = 
	new StringTokenizer(value.toString());
      while (stk.hasMoreTokens()) {
	output.collect(new Text(stk.nextToken()), 
		       new IntWritable(1));
      }
    }
  }
  /* mapperend */
  public static class Reduce extends MapReduceBase 
    implements Reducer<Text, IntWritable, Text, IntWritable> {
    public void reduce(Text key, Iterator<IntWritable> values, 
		       OutputCollector<Text, IntWritable> output, 
		       Reporter reporter)
      throws IOException {
      int sum = 0;
      while(values.hasNext())
	sum += values.next().get();
      output.collect(key, new IntWritable(sum));
    }
  }
  /* reducerend */
  public static void main(String[] args) throws Exception {
    JobConf conf = new JobConf(WordCount.class);
    conf.setJobName("WordCount");

    conf.setMapperClass(Map.class);
    conf.setCombinerClass(Reduce.class);
    conf.setReducerClass(Reduce.class);
    /* setkeyvalue */
    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(IntWritable.class);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(IntWritable.class);
    /* setio */
    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);

    FileInputFormat.setInputPaths(conf, new Path(args[0]));
    FileOutputFormat.setOutputPath(conf, new Path(args[1]));

    JobClient.runJob(conf);
  }
  /* mainend */
}
