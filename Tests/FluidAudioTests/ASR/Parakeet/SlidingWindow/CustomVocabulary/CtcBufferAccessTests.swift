import CoreML
import XCTest

@testable import FluidAudio

/// The spotter reads CTC logits and writes audio through `MLMultiArray` buffers instead of one boxed
/// `NSNumber` per element. These tests hold the buffer paths to the NSNumber paths they replaced, bit for bit.
final class CtcBufferAccessTests: XCTestCase {

    /// Deterministic values with a wide range (and some exact fp16 ties) for both element types.
    private func values(_ n: Int) -> [Float] {
        (0..<n).map { i in Float(sin(Double(i) * 0.731) * 37.5 + Double(i % 7) * 0.000_976_562_5) }
    }

    private func fill(_ array: MLMultiArray, _ vals: [Float]) {
        for i in 0..<vals.count { array[i] = NSNumber(value: vals[i]) }
    }

    private func assertSameBits(_ a: [Float], _ b: [Float], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern), file: file, line: line)
    }

    func testLogitsMatchNSNumberReadsRank3() throws {
        for type in [MLMultiArrayDataType.float32, .float16, .double] {
            let (t, v) = (13, 29)
            let array = try MLMultiArray(shape: [1, NSNumber(value: t), NSNumber(value: v)], dataType: type)
            fill(array, values(t * v))
            let s = array.strides.map(\.intValue)
            let fast = try XCTUnwrap(
                CtcKeywordSpotter.readLogits(array, timeSteps: t, vocabSize: v, timeStride: s[1], vocabStride: s[2]))
            var slow: [Float] = []
            for ti in 0..<t {
                for vi in 0..<v { slow.append(array[[0, ti, vi].map { NSNumber(value: $0) }].floatValue) }
            }
            assertSameBits(fast, slow)
        }
    }

    func testLogitsMatchNSNumberReadsRank4() throws {
        for type in [MLMultiArrayDataType.float32, .float16] {
            let (t, v) = (11, 17)
            let array = try MLMultiArray(
                shape: [1, NSNumber(value: v), 1, NSNumber(value: t)], dataType: type)
            fill(array, values(t * v))
            let s = array.strides.map(\.intValue)
            let fast = try XCTUnwrap(
                CtcKeywordSpotter.readLogits(array, timeSteps: t, vocabSize: v, timeStride: s[3], vocabStride: s[1]))
            var slow: [Float] = []
            for ti in 0..<t {
                for vi in 0..<v { slow.append(array[[0, vi, 0, ti].map { NSNumber(value: $0) }].floatValue) }
            }
            assertSameBits(fast, slow)
        }
    }

    func testSamplesMatchNSNumberWrites() throws {
        let samples = values(1_000).map { $0 / 40 }
        for type in [MLMultiArrayDataType.float32, .float16] {
            for shape in [[NSNumber(value: 1_200)], [1, NSNumber(value: 1_200)]] {
                let fast = try MLMultiArray(shape: shape, dataType: type)
                let slow = try MLMultiArray(shape: shape, dataType: type)
                CtcKeywordSpotter.copySamples(samples, count: 900, into: fast)
                for i in 0..<900 { slow[i] = NSNumber(value: samples[i]) }
                assertSameBits(
                    (0..<1_200).map { fast[$0].floatValue }, (0..<1_200).map { slow[$0].floatValue })
            }
        }
    }
}
